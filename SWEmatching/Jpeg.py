import numpy as np
import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from Utils.utils import *
from Utils.loader import HDQ_loader 
from Utils.args_inputs import *
import argparse
import random
import warnings

# import OptD
import HDQ
# import SDQ
# import SWE_JPEG_d_fixed
# import SWE_OptD_d_fixed
# import SWE_OptD_QF_fixed
# import SWE_OptS_QF_fixed

num_workers=8

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**num_workers
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def main(args):
    fileFormat = args.output_txt
    args.output_txt = fileFormat%(args.QF_Y, args.QF_C)
    BPP, PSNR, top1, top5, loss = running_func(args)
    key = str(args.resize_resl) + "_" + str(args.QF_Y) + "_" + str(args.QF_C)  
    data_file_name =  args.Model+ "_" + str(args.resize_resl)
    write_live("./RESULT_HDQ/"+data_file_name, key, [BPP, PSNR, top1, top5, loss])

        
def running_func(args):
    Batch_size = args.batchsize
    model = args.Model
    J = args.J
    a = args.a
    b = args.b
    QF_Y = args.QF_Y
    QF_C = args.QF_C
    compress_resize = args.compress_resize
    resize_compress = args.resize_compress
    args.comparsionRunner = HDQ.__call__
    # args.comparsionRunner = HDQ.__call__
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print_exp_details(args)
    print("Model: ", model)
    print("Colorspace: ", args.colorspace)
    print("J =", J)
    print("a =", a)
    print("b =", b)
    print("QF_Y =",QF_Y)
    print("QF_C =",QF_C)
    print("Compress_Resize =",compress_resize)
    print("Resize_Compress =",resize_compress)
    print("resize_resl = ", args.resize_resl)
    print("Batch size = ", args.batchsize)

    pretrained_model = load_model(model) 
    _ = pretrained_model.to(device)

    dataset = HDQ_loader(args)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    num_correct = 0
    num_tests = 0
    BPP = 0
    PSNR = 0
    SSIM = 0
    cnt = 0
    loss = 0

    top1 = 0
    top5 = 0
    
    for dt in tqdm.tqdm(test_loader):
        image, image_BPP, image_PSNR , labels = dt
        filter = (image_BPP>-1)
        if (len(image) != filter.sum()):
            breakpoint()
            running_func(args)
        image = image[filter]
        image_PSNR = image_PSNR[filter]
        labels = labels[filter]
        image_BPP = image_BPP[filter]
        if(len(image) == 0):
            continue
        labels = labels.to(device)
        image = image.to(device)
        BPP+=torch.sum(image_BPP)
        PSNR+=torch.sum(image_PSNR)
        pred = pretrained_model(image)
        # breakpoint()
        loss += float(torch.nn.CrossEntropyLoss()(pred, labels))
        num_correct += (pred.argmax(1) == labels).sum().item()
        acc1, acc5  = accuracy(pred, labels, topk=(1, 5)) 
        top1 += acc1
        top5 += acc5
        num_tests += len(labels)
        # if (cnt+1) %100 ==0:
        #     print_iteration(top1, top5, BPP, PSNR, cnt, num_correct, num_tests)
        cnt += 1
    top1 , top5 = top1.cpu().numpy()/num_tests, top5.cpu().numpy()/num_tests
    loss = loss/num_tests
    BPP, PSNR = (BPP.numpy()/num_tests), (PSNR.numpy()/num_tests)
    log_file(top1, top5, BPP, PSNR, loss, cnt, num_correct, num_tests, args.output_txt, args.Model)
    return BPP, PSNR, top1, top5, loss
    

if '__main__' == __name__:
    args = getArgs()
    main(args)