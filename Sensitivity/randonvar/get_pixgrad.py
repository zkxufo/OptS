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

def plot_confidence_interval(x, top, bottom, mean, horizontal_line_width=0.25, color='#2187bb',label=None,alpha=1):
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    plt.plot([x, x], [top, bottom], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [top, top], color=color,alpha=0.7*alpha)
    plt.plot([left, right], [bottom, bottom], color=color,alpha=0.7*alpha)
    plt.plot(x, mean, 'o', color=color, label=label,alpha=alpha)
    return mean
from matplotlib.pyplot import figure

def main(model = 'alexnet', Batch_size = 100, Nexample= 10000, resize = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resize:
        Batch_size = Batch_size
    else:
        Batch_size = 1
    thr = Nexample
    model_name = model
    print("code run on", device)
    resize224 = transforms.Resize((224, 224))
    Trans = [transforms.ToTensor()]
    if resize:
        print("resize")
        Trans.append(transforms.Resize((224, 224)))
    Trans.append(transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]))

    transform = transforms.Compose(Trans)
    dataset = torchvision.datasets.ImageNet(root="~/project/data", split='train',
                                            transform=transform)
    A = load_3x3_weight(model_name).to(device)
    A_inv = torch.linalg.inv(A).to(device)
    # Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pretrained_model = get_model(model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    Y_sen_list = np.empty([0])
    Cr_sen_list = np.empty([0])
    Cb_sen_list = np.empty([0])
    idx = 0
    save_idx = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        # breakpoint()
        img_shape = data.shape[-2:]
        data = data.transpose(0, 1).reshape(3, -1)  # [0,225]
        WXV = A @ (data - 128.)  # [-128, 127]
        WXV.requires_grad = True
        RGB = A_inv @ WXV + 128.  # [0,225]
        # breakpoint()
        recovered_img = RGB.reshape(3, Batch_size, img_shape[0], img_shape[1]).transpose(0, 1)  # [-128, 127]
        # breakpoint()
        norm_img1 = recovered_img/255.  # [0,1]
        norm_img = normalize(norm_img1)
        output = pretrained_model(resize224(norm_img))
        loss = F.nll_loss(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = (WXV.grad).detach().cpu().numpy()
        # breakpoint()
        Y_sen_list = np.concatenate((Y_sen_list, data_grad[0]))
        Cr_sen_list = np.concatenate((Cr_sen_list, data_grad[1]))
        Cb_sen_list = np.concatenate((Cb_sen_list, data_grad[2]))
        # breakpoint()
        if idx*Batch_size >= thr:
            break
        idx += 1
    np.save("./grad/Y_sen_list" + model_name + ".npy", Y_sen_list)
    np.save("./grad/Cr_sen_list" + model_name + ".npy", Cr_sen_list)
    np.save("./grad/Cb_sen_list" + model_name + ".npy", Cb_sen_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    parser.add_argument('-resize', action='store_true', help='Calculate Grad on resize img')
    parser.add_argument('-no-resize', dest='resize', action='store_false', help='Calculate Grad on resize img')
    # parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(**vars(args))
