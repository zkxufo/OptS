import numpy as np
import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import os
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import numpy as np
from io import BytesIO


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, NUM_QFs=0, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=True,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training: 
        images_org = fn.decoders.image_random_crop(images,
                                                device=decoder_device, output_type=types.RGB,
                                                # device_memory_padding=device_memory_padding,
                                                # host_memory_padding=host_memory_padding,
                                                # preallocate_width_hint=preallocate_width_hint,
                                                # preallocate_height_hint=preallocate_height_hint,
                                                random_aspect_ratio=[0.75, 4.0 / 3.0],
                                                random_area=[0.08, 1.0],
                                                num_attempts=100)
        resized_images = fn.resize(images_org,
                            device=dali_device,
                            resize_x=crop,
                            resize_y=crop,
                            interp_type=types.INTERP_TRIANGULAR)
        
        mirror = fn.random.coin_flip(probability=0.5)

        original = fn.crop_mirror_normalize(resized_images.gpu(),
                                        dtype=types.FLOAT,
                                        output_layout="CHW",
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        mirror=mirror)
        
        # List to store all the compressed samples
        compressed_samples = []
        compressed_samples.append(original)
        for qf in np.linspace(start=50, stop=100, num=NUM_QFs, endpoint=False):        
            compressed = fn.jpeg_compression_distortion(
                                                    resized_images.gpu(),
                                                    preserve=True,
                                                    quality=qf,
                                                    seed=0
                                                    )   
            compressed = fn.crop_mirror_normalize(compressed.gpu(),
                                            dtype=types.FLOAT,
                                            output_layout="CHW",
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                            mirror=mirror)
            
            compressed_samples.append(compressed)            
        
        samples = fn.cat(*compressed_samples, axis=0)

    else:
        # This is for validation
        images_org = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        resized_images = fn.resize(
                                images_org,
                                device=dali_device,
                                size=size,
                                mode="not_smaller",
                                interp_type=types.INTERP_TRIANGULAR,
                              )
        mirror = False
        original = fn.crop_mirror_normalize(resized_images.gpu(),
                                        dtype=types.FLOAT,
                                        output_layout="CHW",
                                        crop=(crop, crop),
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                        # mean=[0, 0, 0],
                                        # std=[1, 1, 1],
                                        mirror=mirror)
        # List to store all the compressed samples
        compressed_samples = []
        for qf in np.linspace(start=50, stop=100, num=NUM_QFs, endpoint=False):        
            compressed = fn.jpeg_compression_distortion(
                                                    resized_images.gpu(),
                                                    preserve=True,
                                                    quality=qf,
                                                    seed=0
                                                    )   
            compressed = fn.crop_mirror_normalize(compressed.gpu(),
                                                    dtype=types.FLOAT,
                                                    output_layout="CHW",
                                                    crop=(crop, crop),
                                                    mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                                    std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                                    # mean=[0, 0, 0],
                                                    # std=[1/255, 1/255, 1/255],
                                                    mirror=mirror)
            compressed_samples.append(compressed)      
        
        compressed_samples.append(original)
        samples = fn.cat(*compressed_samples, axis=0)
    labels = labels.gpu()
    return samples, labels


class HRS_Selector(object):
    def __init__(self, batch_size, Split_Batchs, NUM_QFs) -> None:
        self.Split_Batchs = Split_Batchs
        self.BATCH_SIZE = batch_size
        self.NUM_QFs = NUM_QFs
        self.kl_values =  torch.zeros(NUM_QFs+1, self.BATCH_SIZE, dtype=torch.float16).cuda()
        self.selected_samples =  torch.zeros(self.BATCH_SIZE * self.Split_Batchs, 3, 224, 224, dtype=torch.float32).cuda()
        self.original_samples =  torch.zeros(self.BATCH_SIZE * self.Split_Batchs, 3, 224, 224, dtype=torch.float32).cuda()
        self.target =  torch.zeros(self.BATCH_SIZE * self.Split_Batchs,  dtype=torch.int64).cuda()
        # Define the inverse transformations
        # self.normal = transforms.Normalize(mean=[0.485 ,0.456 ,0.406 ], std=[0.229, 0.224, 0.225])
        self.inverse_transform = transforms.Compose([
                                                transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
                                                transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                            ])
        self.transform = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])])
        self.QFs = np.linspace(start=50, stop=100, num=NUM_QFs, endpoint=False)

    def select(self, input, target_minBatch, model, iteration):
        
        data_tensor_window = input.size(0)
        # if the original is repreated once
        split_images = torch.chunk(input, chunks=self.NUM_QFs+1, dim=1)


        # self.inverse_transform(split_images[0][1]).save('reconstructed_image.jpg')
        # np.sum(np.abs(np.array(self.transform(split_images[-1][1]))-np.array(self.transform(split_images[0][1]))))
        # breakpoint()
        # with torch.no_grad(): original = model(self.normal(split_images[-1]))
        with torch.no_grad(): original = model((split_images[-1]))
        _, pred = original.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_minBatch[:data_tensor_window].view(1, -1).expand_as(pred))        
        non_correct = torch.nonzero(torch.eq(correct[0], 0)).squeeze(1)
        entropy =  100 * torch.ones(data_tensor_window)
        entropy[non_correct] = 4999
        self.kl_values[-1, :data_tensor_window] = entropy
        for idx in range(0,len(split_images)-1):
            # compute output
            # with torch.no_grad(): output = model(self.normal(split_images[idx]))
            with torch.no_grad(): output = model((split_images[idx]))
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target_minBatch.view(1, -1).expand_as(pred))        
            kl_divs = self.QFs[idx] * torch.ones(data_tensor_window)
            # Give the miss-classified samples a negative values
            non_correct = torch.nonzero(torch.eq(correct[0], 0)).squeeze(1)     
            kl_divs[non_correct] = 5000
            self.kl_values[idx, :data_tensor_window]  =  kl_divs
        indices =  torch.argmin(self.kl_values[:, :data_tensor_window], axis=0)
        for idx , qf in enumerate(indices):
            self.selected_samples[idx + (iteration%self.Split_Batchs)* self.BATCH_SIZE] = self.transform(split_images[qf][idx])
            self.original_samples[idx + (iteration%self.Split_Batchs)* self.BATCH_SIZE] = self.transform(split_images[-1][idx])
            self.target[idx + (iteration%self.Split_Batchs)* self.BATCH_SIZE] = target_minBatch[idx]

# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from model import get_model
import argparse
class GGD:
    def __init__(self, scale, beta, loc):
        self.scale = scale
        self.loc = loc
        self.beta = beta
    def E(self):
        x = 0
        val = -(self.scale*np.exp(self.loc/self.scale - x/self.scale)^self.beta*(self.scale + self.beta*x))/self.beta^2
        return val

def main(args):
    args.world_size = 1
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    thr = args.Nexample
    model_name = args.model
    args.QFs = 20
    print("code run on", device)
    Trans = [transforms.ToTensor(),
             transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
             ]

    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root="/home/l44ye/DATASETS", split='train',
                                            transform=transform)
    resize256 = transforms.Resize(256)
    CenterCrop224 = transforms.CenterCrop(224)
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.Batch_size, shuffle=True, num_workers=8)

    test_pipe = create_dali_pipeline(batch_size=args.Batch_size,
                                        num_threads=8,
                                        device_id=args.local_rank,
                                        seed=12 + args.local_rank,
                                        data_dir="/home/l44ye/DATASETS/train",
                                        crop=224,
                                        size=256,
                                        NUM_QFs = args.QFs,
                                        dali_cpu=False,
                                        shard_id=args.local_rank,
                                        num_shards=args.world_size,
                                        is_training=False)
    test_pipe.build()
    
    test_loader = DALIClassificationIterator(test_pipe, reader_name="Reader",
                                                    last_batch_policy=LastBatchPolicy.PARTIAL,
                                                    auto_reset=True)
    

    pretrained_model = get_model(model_name)


    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = []
    Cr_sen_list = []
    Cb_sen_list = []
    idx = 0
    criterion = nn.CrossEntropyLoss()
    hkl_selector = HRS_Selector(args.Batch_size, Split_Batchs=1, NUM_QFs=args.QFs)
    iteration = 0
    for data_out in tqdm(test_loader):
        data, target = data_out[0]['data'], data_out[0]['label'].squeeze().cuda().long()
        data, target = data.to(device), target.to(device)  # [0,225]        
        hkl_selector.select(data, target, pretrained_model, iteration)

        data_tensor_window = data.size(0)
        data = hkl_selector.selected_samples[:data_tensor_window * 1]
        # data = hkl_selector.original_samples[:data_tensor_window * 1]
        target = hkl_selector.target[:data_tensor_window * 1]

        img_shape = data.shape[-2:]
        input_DCT_block_batch = block_dct(blockify(rgb_to_ycbcr(data), 8))
        input_DCT_block_batch.requires_grad = True
        recoverd_img = deblockify(block_idct(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
        norm_img = normalize(Scale2One(ycbcr_to_rgb(recoverd_img)))
        output = pretrained_model(norm_img)
        loss = criterion(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = input_DCT_block_batch.grad.transpose(1, 0).detach().cpu().numpy()
        Y = data_grad[0].reshape(-1, 8, 8)
        Y_sen_list.append(Y)
        Cb = data_grad[1].reshape(-1, 8, 8)
        Cb_sen_list.append(Cb)
        Cr = data_grad[2].reshape(-1, 8, 8)
        Cr_sen_list.append(Cr)
        idx += args.Batch_size
        iteration += 1
        if idx >= thr:
            break
    
    # pretrained_model.zero_grad()
    # del loss
    # del input_DCT_block_batch
    # del pretrained_model
    # del recoverd_img
    # del data_grad
    # del Y
    # del Cb
    # del Cr
    
    Y_sen_list = np.array(Y_sen_list).reshape(-1,8,8)
    print("Convert Y")
    Cr_sen_list = np.array(Cr_sen_list).reshape(-1,8,8)
    print("Convert Cr")
    Cb_sen_list = np.array(Cb_sen_list).reshape(-1,8,8)
    print("Convert Cb")
    np.save("./grad/Y_sen_list" + model_name + "_HRS.npy",Y_sen_list)
    print("")
    # del Y_sen_list
    
    np.save("./grad/Cr_sen_list" + model_name + "_HRS.npy", Cr_sen_list)
    # del Cr_sen_list
    np.save("./grad/Cb_sen_list" + model_name + "_HRS.npy", Cb_sen_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-dev',type=str, default='cuda', help='device')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(args)
