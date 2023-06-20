import numpy as np
import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import os
from torch import nn

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
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    thr = args.Nexample
    model_name = args.model
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

    pretrained_model = get_model(model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = []
    Cr_sen_list = []
    Cb_sen_list = []
    idx = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
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
        # Y = data_grad[0].reshape(-1, 8, 8)
        # Y_sen_list.append(Y)
        Cb = data_grad[1].reshape(-1, 8, 8)
        Cb_sen_list.append(Cb)
        # Cr = data_grad[2].reshape(-1, 8, 8)
        # Cr_sen_list.append(Cr)
        idx += args.Batch_size
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
    # Y_sen_list = np.array(Y_sen_list).reshape(-1,8,8)
    # print("Convert Y")
    # Cr_sen_list = np.array(Cr_sen_list).reshape(-1,8,8)
    # print("Convert Cr")
    Cb_sen_list = np.array(Cb_sen_list).reshape(-1,8,8)
    print("Convert Cb")
    # np.save("/home/multicompc15/Documents/DCT_coefficient/grad/Y_sen_list" + model_name + ".npy",Y_sen_list)
    # print("")
    # del Y_sen_list
    
    # np.save("/home/multicompc15/Documents/DCT_coefficient/grad/Cr_sen_list" + model_name + ".npy", Cr_sen_list)
    # del Cr_sen_list
    np.save("./grad/Cb_sen_list" + model_name + ".npy", Cb_sen_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-dev',type=str, default='cuda', help='device')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(args)
