import numpy as np
import torch
from torchvision import models, datasets, transforms
from tqdm import tqdm
from scipy.stats import ttest_ind
from scipy.stats import bootstrap
def get_model(name):
    if name =='alexnet':
        pretrained_model = models.alexnet(pretrained=True)
    elif name == 'resnet':
        pretrained_model = models.resnet18(pretrained=True)
    elif name == 'squeezenet':
        pretrained_model = models.squeezenet1_0(pretrained=True)
    elif name == 'vgg':
        pretrained_model = models.vgg11(pretrained=True)
    return pretrained_model
