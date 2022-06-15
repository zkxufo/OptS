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
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
                                SDQ_transforms(model, QF_Y, QF_C, J, a, b, Lmbd, Beta_S, Beta_W, Beta_X),
                                transforms.Resize((224, 224))])
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
# pretrained_model = models.alexnet(pretrained=True).to(device)
# _ = pretrained_model.eval()
# for idx, dt in enumerate(test_loader):
#     data, tar = dt #[0, 255]
#     norm_img = resize(normalize(data/255.)).type(torch.FloatTensor).to(device)
#     output = pretrained_model(norm_img)
#     init_pred = output.max(1, keepdim=True)[1]
#     tar = tar.to(device)
#     if init_pred==tar:
#         total_counts += 1
#         data = data[0].numpy()#[0, 255]
#         compressed_img, bit_rate = SDQ.__call__(data, "Alexnet", J, a, b,
#                             QF_Y, QF_C, Beta_S, eps)
#         compressed_img = torch.from_numpy((compressed_img/255.)[None])
#         compressed_img=compressed_img.type(torch.FloatTensor)
#         norm_img = resize(normalize(compressed_img)).to(device)
        
#         output = pretrained_model(norm_img)
#         init_pred_comp = output.max(1, keepdim=True)[1]
#         if init_pred_comp==tar:
#             corr_counts +=1
#         print("ACC : ", (corr_counts/(total_counts)*100))
#         print("Correct : ", corr_counts, "  out ", total_counts)
# # i = 0
for dt in tqdm.tqdm(test_loader):
    data_BPP, tar = dt
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

#     break
# plt.imshow(compressed_img.transpose(1,2,0)/255.)#red
# plt.show()
# plt.imshow(data[0]/255.)#red
# plt.show()
# plt.imshow(data[1]/255.)#green
# plt.show()
# plt.imshow(data[2]/255.)#blue
# plt.show()
# plt.imshow(compressed_img.transpose(1,2,0)/255.)
# plt.show() 
