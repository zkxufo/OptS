import numpy as np
import torch
from torchvision import models, datasets, transforms
import torchvision
from tqdm import tqdm
from scipy.stats import ttest_ind
from scipy.stats import bootstrap
from matplotlib.pyplot import figure
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
# import torchattacks
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from utils import *
#from imagenet_class import imagenet_label
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]='1'
Batch_size = 100
from model import get_model
import argparse

from matplotlib.pyplot import figure

def main(model = 'alexnet', Batch_size = 100, Nexample= 10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = Batch_size
    thr = Nexample
    model_name = model
    print("code run on", device)
    Trans = [transforms.ToTensor(),
             transforms.Resize((256, 256)),
             transforms.CenterCrop(224),
             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
             ]
    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root="~/project/data", split='train',
                                            transform=transform)
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True)

    pretrained_model = get_model(model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = np.empty([0, 8, 8])
    Cr_sen_list = np.empty([0, 8, 8])
    Cb_sen_list = np.empty([0, 8, 8])
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        img_shape = data.shape[-2:]
        ycbcr_data = rgb_to_ycbcr(data)
        input_DCT_block_batch = block_dct(blockify(ycbcr_data, 8))
        input_DCT_block_batch.requires_grad = True
        recoverd_img = deblockify(block_idct(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
        recoverd_RGB_img = ycbcr_to_rgb(recoverd_img)
        norm_img1 = Scale2One(recoverd_RGB_img)  # [0,1]
        # breakpoint()
        norm_img = normalize(norm_img1)
        output = pretrained_model(norm_img)
        loss = F.nll_loss(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = torch.mean(torch.abs(input_DCT_block_batch.grad), dim = 2).transpose(1,0).detach().cpu().numpy()
        # breakpoint()
        Y_sen_list = np.concatenate((Y_sen_list, data_grad[0].reshape(-1, 8, 8)))
        Cr_sen_list = np.concatenate((Cr_sen_list, data_grad[1].reshape(-1, 8, 8)))
        Cb_sen_list = np.concatenate((Cb_sen_list, data_grad[2].reshape(-1, 8, 8)))
        if Y_sen_list.shape[0] >= thr:
            break
    np.save("./grad/Y_sen_list" + model_name + ".npy",Y_sen_list)
    np.save("./grad/Cr_sen_list" + model_name + ".npy", Cr_sen_list)
    np.save("./grad/Cb_sen_list" + model_name + ".npy", Cb_sen_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    # parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(**vars(args))
