import numpy as np
# import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
# import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from Utils.utils import *
from Utils.args_inputs import *
from Utils.loader import SWE_matching_loader , HDQ_loader
from perc import Perc

import random
import warnings
import pickle

# import OptD
import HDQ
# import SDQ
# import SWE_JPEG_d_fixed
# import SWE_OptD_d_fixed
import SWE_OptD_QF_fixed
import SWE_OptS_QF_fixed

num_workers=100

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
    # args.machine_dist = False
    print("Machine Distortion enables = ", args.machine_dist)
    base_senstivity =  args.Model
    if args.OptD_enable:
        args.comparsionRunner = SWE_OptD_QF_fixed.__call__
        dir_name = "./Comp1_OptD_QF_fixed/base_senstivity_" + base_senstivity + "/"
    elif args.OptS_enable:
        args.comparsionRunner = SWE_OptS_QF_fixed.__call__
        dir_name = "./Comp1_OptS_QF_fixed/base_senstivity_" + base_senstivity + "/"
    else:
        args.comparsionRunner = HDQ.__call__
        dir_name = "./Comp1_HDQ/"
    print(dir_name.split("/")[1])
    Qmax_flag, BPP, PSNR, top1, top5, loss, pretrained_family_names = running_func(args)
    key  = str(Qmax_flag) + "_" + str(args.QF_Y) + "_" + str(args.QF_C)

    for idx , model_name in enumerate(pretrained_family_names):
        write_live(dir_name+model_name, key, [BPP, PSNR, top1[idx], top5[idx], loss[idx]])

def running_func(args):
    Batch_size = args.batchsize
    model = args.Model
    J = args.J
    a = args.a
    b = args.b
    Qmax_Y = args.Qmax_Y
    Qmax_C = args.Qmax_C
    QF_Y = args.QF_Y
    QF_C = args.QF_C
    DT_Y = args.DT_Y
    DT_C = args.DT_C
    d_waterlevel_Y = args.d_waterlevel_Y
    d_waterlevel_C = args.d_waterlevel_C
    compress_resize = args.compress_resize
    OptD_enable = args.OptD_enable
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print_exp_details(args)
    print("Running Devcie: ", device)
    print("No. of Workers : ", num_workers)
    print("Model: ", args.Model)
    print("Colorspace: ", args.colorspace)
    print("J =", args.J)
    print("a =", args.a)
    print("b =", args.b)
    print("QF_Y:", args.QF_Y)
    print("QF_C:", args.QF_C)
    print("DT_Y:", args.DT_Y)
    print("DT_C:", args.DT_C)
    print("d_waterlevel_Y: ", args.d_waterlevel_Y)
    print("d_waterlevel_C: ", args.d_waterlevel_C)
    print("Qmax_Y =", args.Qmax_Y)
    print("Qmax_C =", args.Qmax_C)
    print("OptD enables =", OptD_enable)
    print("OptS enables =", args.OptS_enable)
    print("HDQ enables =", args.JPEG_enable)

    pretrained_model = []

    for x in list(models_list.keys()):
        if x in args.Model:
            selected_family = x
    pretrained_family_names = models_list[selected_family]
    print("Loaded Models : ", pretrained_family_names)
    for nm in pretrained_family_names: 
        pretrained_model_buff = load_model(nm)
        _ = pretrained_model_buff.to(device)
        pretrained_model.append(pretrained_model_buff)
    
    # pretrained_model = load_model(model) 
    # _ = pretrained_model.to(device)
    
    if args.OptS_enable or args.OptD_enable:
        dataset = SWE_matching_loader(args)
    else:
        dataset = HDQ_loader(args)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    num_tests = 0
    BPP = 0
    PSNR = 0
    SSIM = 0
    cnt = 0
    count_Qmax = 0
    Qmax_flag = False
    
    num_correct =[0] * len(pretrained_model)
    loss = [0] * len(pretrained_model)
    top1 = [0] * len(pretrained_model)
    top5 = [0] * len(pretrained_model)
    
    # for dt in tqdm.tqdm(test_loader):
    for dt in Perc(test_loader):
        image, image_BPP, image_PSNR , labels = dt
        filter = (image_BPP>-1)
        if (len(image) != filter.sum()):
            breakpoint()
            running_func(args)
        if(len(image) == 0):
            continue
        image = image[filter]
        image_PSNR = image_PSNR[filter]
        labels = labels[filter]
        image_BPP = image_BPP[filter]
        count_Qmax += torch.sum(image_BPP < 0)
        image_BPP = torch.abs(image_BPP)
        labels = labels.to(device)
        image = image.to(device)
        BPP+=torch.sum(image_BPP)
        PSNR+=torch.sum(image_PSNR)
        num_tests += len(labels)
        for idx , model in enumerate(pretrained_model): 
            pred = model(image)   
            loss[idx] += float(torch.nn.CrossEntropyLoss()(pred, labels))
            num_correct[idx] += (pred.argmax(1) == labels).sum().item()
            acc1, acc5  = accuracy(pred, labels, topk=(1, 5)) 
            top1[idx] += acc1
            top5[idx] += acc5
        # if (cnt+1) %50 ==0:
        #     print_iteration(top1, top5, BPP, PSNR, cnt, num_correct, num_tests)
        cnt += 1  
    BPP, PSNR = (BPP.numpy()/num_tests), (PSNR.numpy()/num_tests)
    if (count_Qmax == len(dataset)): Qmax_flag = True
    
    for idx , model_name in enumerate(pretrained_family_names):
        top1[idx] , top5[idx] = top1[idx].cpu().numpy()/num_tests, top5[idx].cpu().numpy()/num_tests
        loss[idx] = loss[idx]/num_tests
        log_file(top1[idx], top5[idx], BPP, PSNR, loss[idx], cnt, num_correct[idx], num_tests, args.output_txt, model_name)
    return Qmax_flag, BPP, PSNR, top1, top5, loss, pretrained_family_names

if '__main__' == __name__:
    args = getArgs()
    main(args)