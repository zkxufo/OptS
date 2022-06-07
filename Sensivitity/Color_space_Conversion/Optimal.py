import numpy as np
import torch
from torchvision import models, datasets, transforms
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from matplotlib import pyplot as plt
from utils import get_zigzag, block_dct, block_idct, rgb_to_ycbcr, ycbcr_to_rgb,\
    blockify, deblockify
from model import get_model
import argparse

def main(model = 'alexnet', Batch_size = 100, Nexample= 10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = Batch_size
    thr = Nexample
    model_name = model
    print("code run on", device)
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])])
    dataset = torchvision.datasets.ImageNet(root="~/project/data", split='train',
                                            transform=transform)
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pretrained_model = get_model(model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    sen_list = torch.empty([3, 0])
    cnt = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        data.requires_grad = True
        norm_data = normalize(Scale2One(data))
        output = pretrained_model(norm_data)
        loss = F.nll_loss(output, target)
        loss.backward()
        grad = data.grad.reshape(Batch_size, 3, -1).transpose(1, 0)
        grad = grad.reshape(3, -1).detach().cpu()
        sen_list = torch.cat((sen_list, grad), 1)
        cnt += Batch_size
        if cnt >= thr:
            break

    sen_list = sen_list - torch.mean(sen_list, 1)[:, None]
    eigmat = torch.real(torch.linalg.eig(sen_list@sen_list.transpose(1, 0)).eigenvectors.transpose(1, 0))
    print("Optimal color space conversion matrix for", model_name, "is: ")
    for i in range(3):
        for j in range(3):
            print(eigmat[i, j].item(), " ", end="")
    print(" ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model,  ')
    parser.add_argument('-model',type=str, help='your name, enter it')
    parser.add_argument('-Batch_size', type=int,help='your name, enter it')
    parser.add_argument('-Nexample',type=int, help='your name, enter it')
    args = parser.parse_args()
    main(**vars(args))
