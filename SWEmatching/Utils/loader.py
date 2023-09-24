import torchvision.datasets as datasets
from torchvision import transforms
from Utils.Compress import *
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import numpy as np
import math 

from Utils.utils import *


class resize_compression(object):
    def __init__(self, compressor):
        super().__init__()
        self.compressor = compressor

    def  __call__(self, sample_org):
        sample_org = transforms.Resize(256)(sample_org)
        sample_org = transforms.CenterCrop([224, 224])(sample_org)
        sample = self.compressor(sample_org)
        BPP = sample['BPP']
        sample = sample['image']
        PSNR   = PSNR_cal_RGB(sample_org, sample)

        sample = transforms.ToTensor()(sample)
        sample = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
        return sample, BPP, PSNR

def normal(sample):
    BPP = 0
    PSNR = 0
    sample = transforms.Resize(256)(sample)
    sample = transforms.CenterCrop([224, 224])(sample)
    sample = TF.to_tensor(sample)
    sample = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
    return sample, BPP, PSNR

class HDQ_loader(datasets.ImageNet):
    def __init__(self, args):
        self.split=args.split
        self.root = args.root
        super().__init__(self.root, split=self.split) # val
        if args.OptS_enable:
            self.HDQ_transforms = OptD_transforms(args)
        elif args.JPEG_enable:
            self.HDQ_transforms = HDQ_transforms(args)
        else:
            print("Set one of these flags: <OptS_enable> or <JPEG_enable>")
            exit(0)
        
        if args.resize_compress:
            print("Resize Compress IMAGE ...")
            self.HDQ_preprocess = resize_compression(self.HDQ_transforms)
        else:
            print("NORMAL RAW IMAGE ...")
            self.HDQ_preprocess = normal
        
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample, BPP, PSNR = self.HDQ_preprocess(sample)
        return sample, BPP, PSNR,  target

    def __len__(self):
        return len(self.imgs)


class SWE_matching_loader(datasets.ImageNet):
    def __init__(self, args):
        self.split=args.split
        self.root = args.root
        super().__init__(self.root, split=self.split) # val

        self.HDQ_transforms = SWE_matching_transforms(args)

        if args.resize_compress:
            print("Resize Compress IMAGE ...")
            self.HDQ_preprocess = resize_compression(self.HDQ_transforms)
        else:
            print("NORMAL RAW IMAGE ...")
            self.HDQ_preprocess = normal
        
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample, BPP, PSNR = self.HDQ_preprocess(sample)
        return sample, BPP, PSNR,  target

    def __len__(self):
        return len(self.imgs)
