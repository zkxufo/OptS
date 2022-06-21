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
Batch_size = 1
from model import get_model
import argparse

def plot_confidence_interval(x, top, bottom, mean, horizontal_line_width=0.25, color='#2187bb',label=None,alpha=1):
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    plt.plot([x, x], [top, bottom], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [top, top], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [bottom, bottom], color=color,alpha=0.7*alpha)
    plt.plot(x, mean, 'o', color=color, label=label,alpha=alpha)
    return mean
from matplotlib.pyplot import figure

def main(model = 'Alexnet', Batch_size = 100, Nexample= 10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = 1
    thr = Nexample
    model_name = model
    print("code run on", device)
    Trans = [transforms.ToTensor(),
             transforms.Resize((256, 256)),
             transforms.CenterCrop(224),
             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])]
    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root="~/project/data", split='val',
                                            transform=transform)
    A = load_3x3_weight(model_name).to(device)
    A_inv = torch.linalg.inv(A).to(device)
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pretrained_model = get_model(model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = np.empty([0])
    Cr_sen_list = np.empty([0])
    Cb_sen_list = np.empty([0])
    cnt = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        data = data.transpose(0, 1).reshape(3, -1)  # [0,225]
        # WXV = data
        WXV = A @ (data - 128.)  # [-128, 127]
        WXV.requires_grad = True
        recoverd_img = WXV.reshape(3, Batch_size, 224, 224)  # [-128, 127]
        #################################################
        # recoverd_img = recoverd_img.transpose(1,0)[0]
        # pimg = recoverd_img.detach().cpu().numpy().transpose(1,2,0)
        # breakpoint()
        #################################################
        seq_recoverd_img = recoverd_img.reshape(3, -1)  # [-128, 127]
        seq_recoverd_STD = A_inv @ seq_recoverd_img + 128.  # [0,225]
        recoverd_img_STD = seq_recoverd_STD.reshape(3, Batch_size, 224, 224).transpose(0, 1)
        norm_img1 = Scale2One(recoverd_img_STD)  # [0,1]
        norm_img = normalize(norm_img1)
        output = pretrained_model(norm_img)
        loss = F.nll_loss(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = WXV.grad.detach().cpu().numpy()** 2
        Y_sen_list = np.concatenate((Y_sen_list, data_grad[0]))
        Cb_sen_list = np.concatenate((Cb_sen_list, data_grad[1]))
        Cr_sen_list = np.concatenate((Cr_sen_list, data_grad[2]))
        cnt += Batch_size
        if cnt >= thr:
            break
    Sen = np.array([np.mean(Y_sen_list), np.mean(Cb_sen_list), np.mean(Cr_sen_list)])
    Sen = Sen/np.sum(Sen)
    print(Sen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='Alexnet', help='DNN model')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(**vars(args))
