import numpy as np
import SDQ
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import SDQ
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
        compressed_img, BPP = SDQ.__call__(img, self.model, self.J, self.a, self.b, self.Q,
                                           self.q, self.Beta_S, self.Lambda, 0)
        compressed_img = torch.tensor(compressed_img)
        return compressed_img # {'image': compressed_img, 'BPP': BPP}
