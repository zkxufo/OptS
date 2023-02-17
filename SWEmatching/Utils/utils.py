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





def quantizationTable(QF=50, Luminance=True):
    #  Luminance quantization table
    #  Standard
    # * 16 11 10 16 24  40  51  61
    # * 12 12 14 19 26  58  60  55
    # * 14 13 16 24 40  57  69  56
    # * 14 17 22 29 51  87  80  62
    # * 18 22 37 56 68  109 103 77
    # * 24 35 55 64 81  104 113 92
    # * 49 64 78 87 103 121 120 101
    # * 72 92 95 98 112 100 103 99

    quantizationTableData = np.ones((8, 8), dtype=np.float32)

    if QF == 100:
        # print(quantizationTableData)
        return quantizationTableData

    if Luminance == True:  # Y channel
        quantizationTableData[0][0] = 16
        quantizationTableData[0][1] = 11
        quantizationTableData[0][2] = 10
        quantizationTableData[0][3] = 16
        quantizationTableData[0][4] = 24
        quantizationTableData[0][5] = 40
        quantizationTableData[0][6] = 51
        quantizationTableData[0][7] = 61
        quantizationTableData[1][0] = 12
        quantizationTableData[1][1] = 12
        quantizationTableData[1][2] = 14
        quantizationTableData[1][3] = 19
        quantizationTableData[1][4] = 26
        quantizationTableData[1][5] = 58
        quantizationTableData[1][6] = 60
        quantizationTableData[1][7] = 55
        quantizationTableData[2][0] = 14
        quantizationTableData[2][1] = 13
        quantizationTableData[2][2] = 16
        quantizationTableData[2][3] = 24
        quantizationTableData[2][4] = 40
        quantizationTableData[2][5] = 57
        quantizationTableData[2][6] = 69
        quantizationTableData[2][7] = 56
        quantizationTableData[3][0] = 14
        quantizationTableData[3][1] = 17
        quantizationTableData[3][2] = 22
        quantizationTableData[3][3] = 29
        quantizationTableData[3][4] = 51
        quantizationTableData[3][5] = 87
        quantizationTableData[3][6] = 80
        quantizationTableData[3][7] = 62
        quantizationTableData[4][0] = 18
        quantizationTableData[4][1] = 22
        quantizationTableData[4][2] = 37
        quantizationTableData[4][3] = 56
        quantizationTableData[4][4] = 68
        quantizationTableData[4][5] = 109
        quantizationTableData[4][6] = 103
        quantizationTableData[4][7] = 77
        quantizationTableData[5][0] = 24
        quantizationTableData[5][1] = 35
        quantizationTableData[5][2] = 55
        quantizationTableData[5][3] = 64
        quantizationTableData[5][4] = 81
        quantizationTableData[5][5] = 104
        quantizationTableData[5][6] = 113
        quantizationTableData[5][7] = 92
        quantizationTableData[6][0] = 49
        quantizationTableData[6][1] = 64
        quantizationTableData[6][2] = 78
        quantizationTableData[6][3] = 87
        quantizationTableData[6][4] = 103
        quantizationTableData[6][5] = 121
        quantizationTableData[6][6] = 120
        quantizationTableData[6][7] = 101
        quantizationTableData[7][0] = 72
        quantizationTableData[7][1] = 92
        quantizationTableData[7][2] = 95
        quantizationTableData[7][3] = 98
        quantizationTableData[7][4] = 112
        quantizationTableData[7][5] = 100
        quantizationTableData[7][6] = 103
        quantizationTableData[7][7] = 99
    else:
        # Standard Cb Cr channel
        # 17 18  24  47  99  99  99  99
        # 18 21  26  66  99  99  99  99
        # 24 26  56  99  99  99  99  99
        # 47 66  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99

        quantizationTableData[0][0] = 17
        quantizationTableData[0][1] = 18
        quantizationTableData[0][2] = 24
        quantizationTableData[0][3] = 47
        quantizationTableData[0][4] = 99
        quantizationTableData[0][5] = 99
        quantizationTableData[0][6] = 99
        quantizationTableData[0][7] = 99
        quantizationTableData[1][0] = 18
        quantizationTableData[1][1] = 21
        quantizationTableData[1][2] = 26
        quantizationTableData[1][3] = 66
        quantizationTableData[1][4] = 99
        quantizationTableData[1][5] = 99
        quantizationTableData[1][6] = 99
        quantizationTableData[1][7] = 99
        quantizationTableData[2][0] = 24
        quantizationTableData[2][1] = 26
        quantizationTableData[2][2] = 56
        quantizationTableData[2][3] = 99
        quantizationTableData[2][4] = 99
        quantizationTableData[2][5] = 99
        quantizationTableData[2][6] = 99
        quantizationTableData[2][7] = 99
        quantizationTableData[3][0] = 47
        quantizationTableData[3][1] = 66
        quantizationTableData[3][2] = 99
        quantizationTableData[3][3] = 99
        quantizationTableData[3][4] = 99
        quantizationTableData[3][5] = 99
        quantizationTableData[3][6] = 99
        quantizationTableData[3][7] = 99
        quantizationTableData[4][0] = 99
        quantizationTableData[4][1] = 99
        quantizationTableData[4][2] = 99
        quantizationTableData[4][3] = 99
        quantizationTableData[4][4] = 99
        quantizationTableData[4][5] = 99
        quantizationTableData[4][6] = 99
        quantizationTableData[4][7] = 99
        quantizationTableData[5][0] = 99
        quantizationTableData[5][1] = 99
        quantizationTableData[5][2] = 99
        quantizationTableData[5][3] = 99
        quantizationTableData[5][4] = 99
        quantizationTableData[5][5] = 99
        quantizationTableData[5][6] = 99
        quantizationTableData[5][7] = 99
        quantizationTableData[6][0] = 99
        quantizationTableData[6][1] = 99
        quantizationTableData[6][2] = 99
        quantizationTableData[6][3] = 99
        quantizationTableData[6][4] = 99
        quantizationTableData[6][5] = 99
        quantizationTableData[6][6] = 99
        quantizationTableData[6][7] = 99
        quantizationTableData[7][0] = 99
        quantizationTableData[7][1] = 99
        quantizationTableData[7][2] = 99
        quantizationTableData[7][3] = 99
        quantizationTableData[7][4] = 99
        quantizationTableData[7][5] = 99
        quantizationTableData[7][6] = 99
        quantizationTableData[7][7] = 99

    if QF >= 1:
        if QF < 50:
            S = 5000 / QF
        else:
            S = 200 - 2 * QF

        for i in range(8):
            for j in range(8):
                q = (50 + S * quantizationTableData[i][j]) / 100
                q = np.clip(np.floor(q), 1, 255)
                quantizationTableData[i][j] = q
    return quantizationTableData

