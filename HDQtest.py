import numpy as np
import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from utils import load_model
from Compress import HDQ_transforms
import argparse
def main(args):
    Batch_size = 1
    model = args.Model
    J = args.J
    a = args.a
    b = args.b
    QF_Y = args.QF_Y
    QF_C = args.QF_C
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Model: ", model)
    print("J =", J)
    print("a =", a)
    print("b =", b)
    print("QF_Y =",QF_Y)
    print("QF_C =",QF_C)
    # pretrained_model = models.vgg11(pretrained=True)
    # pretrained_model = models.resnet18(pretrained=True)
    # pretrained_model = models.squeezenet1_0(pretrained=True)
    # pretrained_model = models.alexnet(pretrained=True)
    pretrained_model = load_model(model) 
    _ = pretrained_model.to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256, 256)),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
                                    HDQ_transforms(QF_Y, QF_C, J, a, b),
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
        # if (cnt+1) %1000 ==0:
        #     print(num_correct/num_tests,"=",num_correct,"/",num_tests)
        #     print(BPP/num_tests)
        # cnt += 1
    print(num_correct/num_tests,"=",num_correct,"/",num_tests)
    print(BPP/num_tests)
if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="HDQ")
    parser.add_argument('--Model', type=str, default="Alexnet", help='Subsampling b')
    parser.add_argument('--J', type=int, default=4, help='Subsampling J')
    parser.add_argument('--a', type=int, default=4, help='Subsampling a')
    parser.add_argument('--b', type=int, default=4, help='Subsampling b')
    parser.add_argument('--QF_Y', type=int, default=50, help='Quality factor of Y channel')
    parser.add_argument('--QF_C', type=int, default=50, help='Quality factor of Cb & Cr channel')
    args = parser.parse_args()
    main(args)
