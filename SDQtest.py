from curses import savetty
import numpy as np
import SDQ
import tqdm
import time
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
from Compress import SDQ_transforms

Batch_size = 1
J = 4
a = 4
b = 4
QF_Y = 50
QF_C = 50
Beta_S = 1000
Beta_W = 1000
Beta_X = 1000
Lmbd = 1
model = "Alexnet"
eps = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# pretrained_model = models.vgg11(pretrained=True)
# pretrained_model = models.resnet18(pretrained=True)
# pretrained_model = models.squeezenet1_0(pretrained=True)
pretrained_model = models.alexnet(pretrained=True)
_ = pretrained_model.eval().to(device)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
                                SDQ_transforms(model, QF_Y, QF_C, J, a, b, Lmbd, Beta_S, Beta_W, Beta_X),
                                ])
dataset = datasets.ImageNet(root="~/project/data", split='val', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=6)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
num_correct = 0
num_tests = 0
BPP = 0
cnt = 0
for dt in tqdm.tqdm(test_loader):
    data_BPP, labels = dt
    labels = labels.to(device)
    resizedimg = data_BPP['image'].to(device)/255.
    normdata = normalize(resizedimg)
    pred = pretrained_model(normdata)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    BPP+=data_BPP['BPP']
    if (cnt+1) %1000 ==0:
        print(num_correct/num_tests,"=",num_correct,"/",num_tests)
        print(BPP/num_tests)
    cnt += 1
print(num_correct/num_tests,"=",num_correct,"/",num_tests)
print(BPP/num_tests)
