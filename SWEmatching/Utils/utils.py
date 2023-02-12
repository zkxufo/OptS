import torch 
from torchvision import models
import os
import numpy as np
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def SSIM_cal_RGB(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(img1, dtype=np.float32)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def PSNR_cal_RGB(s, r, max_value=255):
    s = np.array(s, dtype=np.float64) 
    r = np.array(r, dtype=np.float64)
    
    height, width, channel = np.shape(s)
    size = height*width 

    sb,sg,sr = cv2.split(s)
    rb,rg,rr = cv2.split(r)
    
    
    mseb = ((sb-rb)**2).sum()
    mseg = ((sg-rg)**2).sum()
    mser = ((sr-rr)**2).sum()
    mse = (mseb+mseg+mser)
    
    if mse == 0:
        return float('inf')
    
    MSE = mse/(3*size)
    psnr = 10*math.log10(255**2/MSE)
    return round(psnr,2)

def PSNR_cal_Y(s, r, max_value=255):
    s = np.array(s, dtype=np.float32) 
    r = np.array(r, dtype=np.float32) 
    height, width, channel = np.shape(s)
    size = height*width 
    
    # sb,sg,sr = cv2.split(s)
    s = cv2.cvtColor(s, cv2.COLOR_RGB2YUV)
    r = cv2.cvtColor(r, cv2.COLOR_RGB2YUV)

    sy,su,sv = cv2.split(s)
    ry,ru,rv = cv2.split(r)
    
    msey = ((sy-ry)**2).sum()/size
    mseu = ((su-ru)**2).sum()/size
    msev = ((sv-rv)**2).sum()/size
    
    psnr_y = 10*math.log10(255**2/msey)
    psnr_u = 10*math.log10(255**2/mseu)
    psnr_v = 10*math.log10(255**2/msev)
    
    psnr = psnr_y
    return round(psnr,2)
    
def PSNR_cal_YUV(s, r, max_value=255):
    s = np.array(s, dtype=np.float32) 
    r = np.array(r, dtype=np.float32) 
    height, width, channel = np.shape(s)
    size = height*width 
     
    s = cv2.cvtColor(s, cv2.COLOR_RGB2YUV)
    r = cv2.cvtColor(r, cv2.COLOR_RGB2YUV)

    sy,su,sv = cv2.split(s)
    ry,ru,rv = cv2.split(r)
    
    msey = ((sy-ry)**2).sum()/size
    mseu = ((su-ru)**2).sum()/size
    msev = ((sv-rv)**2).sum()/size
    
    psnr_y = 10*math.log10(255**2/msey)
    psnr_u = 10*math.log10(255**2/mseu)
    psnr_v = 10*math.log10(255**2/msev)
    
    psnr  = (6 * psnr_y + psnr_u + psnr_v) / 8  
    
    return round(psnr,2)

def load_model(Model, device="cpu"):
    if Model=="Alexnet":
        pretrained_model = models.alexnet(pretrained=True).eval()
    elif Model == 'mobilenet_v2':
        pretrained_model = models.mobilenet_v2(pretrained=True)
    else: 
        print("Enter a model SOS")
        exit(0)
    pretrained_model = pretrained_model.eval()
    return pretrained_model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k)
    return res

def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])

        result_list = []
        for k in topk:
            # print(corrects[:k].flatten().sum(dtype=torch.float32), batch_size)
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            # result_list.append(correct_k * (100.0 / batch_size))
            result_list.append(correct_k)
        return result_list

def creat_dir(output_txt):
    sub_dir = output_txt.rsplit("/", 1)[0]
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir) 

def print_file(l1, output_txt):
    # need to add folder to this file
    if output_txt=="": 
        return 0
    creat_dir(output_txt)
    print(l1)
    f = open(output_txt, "+a")
    f.write(l1)
    f.close()

def print_exp_details(args):
    creat_dir(args.output_txt)
    f = open(args.output_txt, "+a")
    f.write(args.device+ "\n")
    f.write("Model: "+ args.Model+ "\n")
    f.write("ColorSpace: "+ str(args.colorspace)+ "\n")
    f.write("J = "+ str(args.J)+ "\n")
    f.write("a = "+ str(args.a)+ "\n")
    f.write("b = "+ str(args.b)+ "\n")
    f.write("QF_Y = "+ str(args.QF_Y)+ "\n")
    f.write("QF_C = "+ str(args.QF_C)+ "\n")
    f.close()

def write_live(filename, key, vec):
    creat_dir(filename +'.txt')
    f = open(filename +'.txt', "+a")
    f.write(key + "\t")
    for x in vec:
        f.write(str(x)+ "\t")
    f.write("\n")
    f.close()
    
def print_iteration(top1, top5, BPP, PSNR, cnt, num_correct, num_tests, model=""):
    l0 = "--> " + str(num_tests) + "\n"
    l1 = model + " --> " + str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
    l2 = str(BPP.numpy()/num_tests) + "\t" + str(PSNR.numpy()/num_tests) + "\t" + str((top1.cpu().numpy()/num_tests)) + "\t" + str((top5.cpu().numpy()/num_tests)) + "\n"
    l = l0 + l1 + l2
    print(l)
    # print_file(l, output_txt)
    
def log_file(top1, top5, BPP, PSNR, loss, cnt, num_correct, num_tests, output_txt, model=""):
    l0 = "#"* 50 + "\n" + "Total Number Images --> " + str(num_tests) + "\n"
    l1 = model + " --> " +str(BPP) + "\t"+ str(PSNR) + "\t"
    l2 = str(top1) + "\t" + str(top5) + "\n"  + "average loss = "+ str(loss) + "\n"
    l = l0 + l1 + l2
    print(l)
    print_file(l, output_txt)





