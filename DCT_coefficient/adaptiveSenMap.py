import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.autograd.functional import jacobian
import math
from utils import *

test_senmap = np.array([2.2127e-06, 2.9158e-06, 2.7789e-06, 3.0015e-06, 2.8771e-06, 3.1155e-06,
                        3.2429e-06, 2.8764e-06, 2.8229e-06, 2.9747e-06, 3.0755e-06, 3.1713e-06,
                        3.1808e-06, 3.1889e-06, 3.1714e-06, 3.7346e-06, 2.9661e-06, 3.3554e-06,
                        3.1444e-06, 2.5626e-06, 3.3753e-06, 3.1565e-06, 2.5307e-06, 3.2048e-06,
                        3.4006e-06, 3.3024e-06, 2.4954e-06, 2.9446e-06, 1.8663e-06, 1.9987e-06,
                        2.4917e-06, 2.3011e-06, 2.4824e-06, 2.2741e-06, 2.3518e-06, 1.9936e-06,
                        1.2480e-06, 1.8267e-06, 1.9741e-06, 1.4916e-06, 1.5035e-06, 1.5623e-06,
                        1.0751e-06, 1.0078e-06, 1.1463e-06, 1.0773e-06, 1.2726e-06, 1.7099e-06,
                        8.8211e-07, 7.2222e-07, 9.9623e-07, 8.8266e-07, 8.9607e-07, 6.5845e-07,
                        4.8956e-07, 7.3514e-07, 7.3342e-07, 4.8534e-07, 3.7399e-07, 5.1439e-07,
                        4.0520e-07, 3.0960e-07, 2.8731e-07, 2.2146e-07])

def C(i):
    if i == 0:
        return 1/(2*math.sqrt(2))
    else:
        return 1/2
def getDCTCoef(size = (8,8)):
    DCT_coef = torch.zeros(size).double()
    for i in range(size[0]):
        for k in range(size[1]):
            C_i = C(i)
            DCT_coef[i,k]=C_i*math.cos((2*k+1)*i*math.pi/16)
    return DCT_coef
DCT_coef = getDCTCoef((8,8))
IDCT_coef = torch.inverse(DCT_coef)

def IDCT(blocks):
    return IDCT_coef@(blocks[None,None,:])@IDCT_coef.transpose(1,0)
def DCT(blocks):
    return DCT_coef@(blocks[None,None,:])@DCT_coef.transpose(1,0)

def CenterPad(blocks, size):
    if size == (8,8):
        return blocks
    padded_block = torch.zeros(size)
    ex_row = size[0]-8
    ex_col = size[1]-8
    N_left = ex_row//2
    N_right = ex_row-N_left
    N_up = ex_col//2
    N_down = ex_col-N_up
    padded_block[N_left:-N_right,N_up:-N_down] = blocks
    return padded_block
resize8 = transforms.Resize(8)
def process(DCTblock):
    SP_Block = IDCT(DCTblock)
    padded_block = CenterPad(SP_Block, (11,11))
    resized_block = resize8(padded_block[None,None,:].double())
    return DCT(resized_block)[0,0]

radblocks = torch.tensor(np.random.rand(8,8)).double()
IDCTjacobian = jacobian(process, radblocks)[0,0]

zigzag = get_zigzag()
flat_J = np.zeros((64,64))
for i in range(8):
    for j in range(8):
        for k in range(8):
            for v in range(8):
                x = int(zigzag[i,j])
                y = int(zigzag[k,v])
                flat_J[x, y] = np.abs(IDCTjacobian[i,j,k,v])
adpSenMap = flat_J.T@test_senmap
plt.plot(adpSenMap)
plt.show()
