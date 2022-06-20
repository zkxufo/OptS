# HDQtest.py

# MIT License

# Copyright (c) 2022 deponce(Linfeng Ye), University of Waterloo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from bdb import Breakpoint
from curses import savetty
import numpy as np
import tqdm
import time
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
from Compress import HDQ_transforms

Batch_size = 1
J = 4
a = 2
b = 0
QF_Y = 25
QF_C = 25
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# pretrained_model = models.vgg11(pretrained=True)
# pretrained_model = models.resnet18(pretrained=True)
# pretrained_model = models.squeezenet1_0(pretrained=True)
pretrained_model = models.alexnet(pretrained=True).to(device)
_ = pretrained_model.eval()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
                                HDQ_transforms(QF_Y, QF_C, J, a, b),
                                ])
# resize = transforms.Resize((256, 256))
# crop = transforms.CenterCrop(224)
dataset = datasets.ImageNet(root="~/project/data", split='val', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=6)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
num_correct = 0
num_tests = 0
BPP = 0
cnt = 0
for dt in tqdm.tqdm(test_loader):
    data_BPP, labels = dt
    # resizedimg = crop(resize(data_BPP['image']))/255
    resizedimg = data_BPP['image']/255
    normdata = normalize(resizedimg)
    pred = pretrained_model(normdata)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    BPP+=data_BPP['BPP']
    if cnt %1000 ==0:
        print(num_correct/num_tests,"=",num_correct,"/",num_tests)
        print(BPP/num_tests)
    cnt += 1
print(num_correct/num_tests,"=",num_correct,"/",num_tests)
print(BPP/num_tests)
