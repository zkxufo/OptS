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
a = 1
b = 0
QF_Y = 50
QF_C = 50
Beta_S = 1
Beta_W = 1
Beta_X = 1
Lmbd = 8
model = "NoModel"
eps = 10
# pretrained_model = models.alexnet(pretrained=True).to(device)
# _ = pretrained_model.eval()
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
                                HDQ_transforms(QF_Y, QF_C, J, a, b),
                                ])
resize = transforms.Resize((224, 224))
dataset = datasets.ImageNet(root="~/project/data", split='val', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=6)

# # corr_counts = 0
# # total_counts = 0
# data = np.array(Image.open(r"./sample/img101.jpeg")).transpose(2,0,1)
# data = np.array(Image.open(r"./sample/ILSVRC2012_val_00017916.JPEG")).transpose(2,0,1)
# ILSVRC2012_val_00017916.JPEG
# plt.imshow(data.transpose(1,2,0)/255.)
# plt.show()
# compressed_img, bit_rate = SDQ.__call__(data, "NoModel", J, a, b,
#                                         QF_Y, QF_C, Beta_S, Lmbd, eps)

# plt.imshow(compressed_img.transpose(1,2,0)/255.)
# plt.show()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for dt in tqdm.tqdm(test_loader):
    data_BPP, tar = dt
    resizedimg = resize(data_BPP['image'])
    print("BPP: ",data_BPP['BPP'])
    # breakpoint()
    # data = data_BPP['image']
    # BPP = data_BPP['BPP']
    # data = data[0].numpy()#[0, 255]
    # plt.imshow(data.transpose(1,2,0)/255.)
    # plt.show()
    # compressed_img, BPP = SDQ.__call__(data, model, J, a, b,
    #                                         QF_Y, QF_C, Beta_S, Lmbd, eps)
    # breakpoint()

    # plt.imshow(data_BPP[0].numpy().transpose(1, 2, 0) / 255.)
    # plt.show()
    # print(BPP)
