# Compress.py

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

import numpy as np
import SDQ
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import HDQ
import matplotlib.pyplot as plt
from PIL import Image

class SDQ_transforms(torch.nn.Module):
    def __init__(self, model="NoModel", Q=50, q=50, J=4, a=4, b=4,
                 Lambda=1, Beta_S=1,Beta_W=1,Beta_X=1,):

        self.model = model
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.Lambda = Lambda
        self.Beta_S = Beta_S
        self.Beta_W = Beta_W
        self.Beta_X = Beta_X
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = SDQ.__call__(img, self.model, self.J, self.a, self.b, 
                                           self.Q, self.q, self.Beta_S, self.Lambda, 0.)
        compressed_img = torch.tensor(compressed_img)
        return{'image': compressed_img, 'BPP': BPP}

class HDQ_transforms(torch.nn.Module):
    def __init__(self, Q=50, q=50, J=4, a=4, b=4):
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = HDQ.__call__(img, self.J, self.a, self.b,
                                           self.Q,self.q)
        compressed_img = torch.tensor(compressed_img)
        return {'image': compressed_img, 'BPP': BPP}
