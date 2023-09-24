import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import os
import sys    
from Utils.utils import quantizationTable


from Utils.utils import *
import matplotlib.pyplot as plt
from PIL import Image

def normalize(arr, factor):
    if factor == 0:
        factor = 10*np.max(arr)
    arr = arr/factor
    return arr, factor


def colorSpace_machineSenstivity(args):
    ColorSpaceW =  np.array( [
                                [  0.299, 0.587, 0.114 ],
                                [ -0.168736, -0.331264, 0.5  ],
                                [ 0.5, -0.418688, -0.081312]
                             ], dtype=float)
    InvColorSpaceW = np.linalg.inv(ColorSpaceW).astype(float)
    
    eps = 1e-5
    InvColorSpaceW[np.abs(InvColorSpaceW) < eps] = 0 # adjust the weight matrix in the color matrix
    sen_map = np.ones((3,64), dtype=float)
    
    print("Sensitivity is loaded :D --> ", args.SenMap_dir+args.Model)

    sen_map[0], factor   = normalize(np.load(args.SenMap_dir+"Y" +args.Model+".npy"), 0 )
    sen_map[1],  _       = normalize(np.load(args.SenMap_dir+"Cb"+args.Model+".npy"), factor)
    sen_map[2],  _       = normalize(np.load(args.SenMap_dir+"Cr"+args.Model+".npy"), factor)
    
    BiasPerImageFlag = False
    return ColorSpaceW, InvColorSpaceW, sen_map, BiasPerImageFlag

class HDQ_transforms(object):
    def __init__(self, args):
    # , model="VGG11", Q=50, q=50, J=4, a=4, b=4):
        self.QF_Y = args.QF_Y
        self.QF_C = args.QF_C
        self.J = args.J
        self.a = args.a
        self.b = args.b
        self.model = args.Model
        self.ColorSpaceW, self.InvColorSpaceW, _ , self.BiasPerImageFlag = colorSpace_machineSenstivity(args)
        self.comparsionRunner = args.comparsionRunner
    def __call__(self, sample):
        sample = np.asarray(sample, dtype=float)
        sample = np.transpose(sample, (2,0,1))
        compressed_img, BPP = self.comparsionRunner( sample, 
                                            self.ColorSpaceW,
                                            self.InvColorSpaceW,
                                            self.BiasPerImageFlag,
                                            self.J, self.a, self.b,
                                            self.QF_Y, self.QF_C)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img) 
        compressed_img = np.transpose(compressed_img, (1,2,0))
        return {'image': compressed_img, 'BPP': BPP}

    def __repr__(self):
        return 'HDQ_transforms!'

class OptD_transforms(object):
    def __init__(self, args): 
        self.J = args.J
        self.a = args.a
        self.b = args.b
        self.model = args.Model
        self.DT_Y = args.DT_Y
        self.DT_C = args.DT_C
        self.d_waterlevel_Y = args.d_waterlevel_Y
        self.d_waterlevel_C = args.d_waterlevel_C
        self.Qmax_Y = args.Qmax_Y
        self.Qmax_C = args.Qmax_C
        self.comparsionRunner = args.comparsionRunner
        self.ColorSpaceW, self.InvColorSpaceW, self.sen_map, self.BiasPerImageFlag = colorSpace_machineSenstivity(args)
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = np.transpose(sample, (2,0,1))
        compressed_img, q_table, BPP = self.comparsionRunner(sample, self.sen_map, 
                                        self.ColorSpaceW,
                                        self.InvColorSpaceW,
                                        self.BiasPerImageFlag,
                                        self.J, self.a, self.b,
                                        self.DT_Y, self.DT_C, 
                                        self.d_waterlevel_Y, self.d_waterlevel_C, 
                                        self.Qmax_Y, self.Qmax_C)

        

        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img)
        compressed_img = np.transpose(compressed_img, (1,2,0))
        return {'image': compressed_img, 'BPP': BPP}


class SWE_matching_transforms(object):
    def __init__(self, args): 
        self.J = args.J
        self.a = args.a
        self.b = args.b
        self.model = args.Model
        self.DT_Y = args.DT_Y
        self.DT_C = args.DT_C
        self.d_waterlevel_Y = args.d_waterlevel_Y
        self.d_waterlevel_C = args.d_waterlevel_C
        self.Qmax_Y = args.Qmax_Y
        self.Qmax_C = args.Qmax_C
        self.QF_Y = args.QF_Y
        self.QF_C = args.QF_C
        self.comparsionRunner = args.comparsionRunner
        self.ColorSpaceW, self.InvColorSpaceW, self.sen_map, self.BiasPerImageFlag = colorSpace_machineSenstivity(args)
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = np.transpose(sample, (2,0,1))
        compressed_img, q_table, BPP =  self.comparsionRunner(sample, self.sen_map, 
                                        self.ColorSpaceW,
                                        self.InvColorSpaceW,
                                        self.BiasPerImageFlag,
                                        self.J, self.a, self.b, self.QF_Y, self.QF_C,
                                        self.DT_Y, self.DT_C, 
                                        self.d_waterlevel_Y, self.d_waterlevel_C, 
                                        self.Qmax_Y, self.Qmax_C)

        
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img)
        compressed_img = np.transpose(compressed_img, (1,2,0))

        return {'image': compressed_img, 'BPP': BPP}

