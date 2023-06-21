import numpy as np
import matplotlib.pyplot as plt
import os
# from results_fromFiles import *

# plt.style.use('ggplot')
# plt.style.use('bmh')

plt.rcParams["font.family"] = "Times New Roman"
font = 25
# Set the default text font size
# plt.rc('font', size=14, weight='bold')
plt.rc('figure', figsize = (8,6))

plt.rc('font', size=font+4)
# Set the axes title font size
plt.rc('axes', titlesize=font)
# Set the axes labels font size
plt.rc('axes', labelsize=font)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=font)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=font)
# Set the legend font size
plt.rc('legend', fontsize=15)
# Set the font size of the figure title
plt.rc('figure', titlesize=font)


def parseLine(L):
    bpp = float(L.split("\t")[1])
    acc = float(L.split("\t")[3])
    return bpp, acc


def CONCAVE(BPPJ, ACCJ):
    BPPJ = np.array(BPPJ)
    ACCJ = np.array(ACCJ) 
    acc_idx = np.argsort(ACCJ, axis=0)[::-1]
    acc = []
    bbp = []
    count = 0
    for idx in acc_idx:
        if len(acc) == 0: 
            acc.append(ACCJ[idx])
            bbp.append(BPPJ[idx])
            count += 1
            continue
        if(BPPJ[idx] < bbp[count-1]):
            acc.append(ACCJ[idx])
            bbp.append(BPPJ[idx])
            count += 1
    return bbp, acc


def CONCAVE_LOWER(BPPJ, ACCJ):
    BPPJ = np.array(BPPJ)
    ACCJ = np.array(ACCJ) 
    acc_idx = np.argsort(ACCJ, axis=0)
    acc = []
    bbp = []
    count = 0
    for idx in acc_idx:
        if len(acc) == 0: 
            acc.append(ACCJ[idx])
            bbp.append(BPPJ[idx])
            count += 1
            continue
        if(BPPJ[idx] > bbp[count-1]):
            acc.append(ACCJ[idx])
            bbp.append(BPPJ[idx])
            count += 1
    # print(acc)
    # print(bbp)
    # breakpoint()
    return bbp, acc

def find_nearest(array, value, min_arg=0):
    array = np.asarray(array)
    # idx = (np.abs(array - value)).argmin()
    idx = np.argsort(np.abs(array-value))[min_arg]
    return array[idx], idx

def select_points(bbp, acc, BPP, ACC):
    FLAG = False
    count = 0
    while(count < np.size(BPP)):
        bbp_, idx_ = find_nearest(BPP, value=bbp, min_arg=count)
        count += 1
        if (BPP[idx_] < bbp) and (ACC[idx_] > acc):
            FLAG = True
    return FLAG

def select_points_reverse(bbp, acc, BPP, ACC):
    FLAG = False
    count = 0
    while(count < np.size(BPP)):
        bbp_, idx_ = find_nearest(BPP, value=bbp, min_arg=count)
        count += 1
        if (BPP[idx_] > bbp) and (ACC[idx_] < acc):
            FLAG = True
    return FLAG

def load_curve(OptS_dir, base_senstivity_dir, JPEG_dir, acclb = 55, bppup=4):
    with open(OptS_dir) as f:
        lines = f.readlines()
    BPPG = []
    ACCG = []
    for L in lines:
        bpp, acc = parseLine(L)
        if bpp == 0: continue
        if not acc < acclb:
            if bpp < bppup:
                BPPG.append(bpp)
                ACCG.append(acc)
    BPPG, ACCG = CONCAVE(BPPG, ACCG)  


    with open(base_senstivity_dir) as f:
        lines = f.readlines()
    BPP = []
    ACC = []
    for L in lines:
        bpp, acc = parseLine(L)
        if bpp == 0: continue
        # FLAG_output = select_points(bpp, acc, BPPG, ACCG)
        FLAG_output = True
        if not acc < acclb and FLAG_output:
            if bpp < bppup:
                BPP.append(bpp)
                ACC.append(acc)

    BPP, ACC = CONCAVE(BPP, ACC)   

    with open(JPEG_dir) as f:
        lines = f.readlines()
    BPPJ = []
    ACCJ = []
    for L in lines:
        bpp, acc = parseLine(L)
        if bpp == 0: continue
        # FLAG_output = select_points(bpp, acc, BPPG, ACCG)
        FLAG_output = True
        if not acc < acclb and FLAG_output:
            if bpp < bppup:
                BPPJ.append(bpp)
                ACCJ.append(acc)

    BPPJ, ACCJ = CONCAVE_LOWER(BPPJ, ACCJ)
    # BPPJ, ACCJ = CONCAVE(BPPJ, ACCJ)

    BPPG_ = []
    ACCG_ = []
    for idx, value in enumerate(BPPG):
        FLAG_output = select_points_reverse(BPPG[idx], ACCG[idx], BPP, ACC) or select_points_reverse(BPPG[idx], ACCG[idx], BPPJ, ACCJ)
        if FLAG_output:
            BPPG_.append(BPPG[idx])
            ACCG_.append(ACCG[idx])
    # BPPG_ = BPPG
    # ACCG_ = ACCG
    ACC = np.array(ACC)*100
    ACCG_ = np.array(ACCG_)*100
    ACCJ = np.array(ACCJ)*100
    return {"OPTD":(BPP, ACC), "OPTDG":(BPPG_, ACCG_), "JPEG":(BPPJ, ACCJ)}

models_list = {

    "Resnet": ["Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152"],
    
    "convnex": ["convnext_base", "convnext_tiny", "convnext_large", 
                "convnext_small"],
    
    "Regnet": ["Regnet400mf", "Regnet800mf", "Regnet2gf", "Regnet6gf",
            #    "Regnet8gf", "Regnet16gf", "Regnet32gf"
               ],
}

bounds = {
          "Regnet400mf": [11, 0],
          "Regnet800mf": [11, 0],
          "Regnet2gf":   [11, 0],
          "Regnet6gf":   [11, 0],
         }

for model in ["Regnet400mf", "Regnet800mf", "Regnet2gf", "Regnet6gf"]:
    JPEG_dir =  "../SWEmatching/Comp1_HDQ/{}.txt".format(model)
    OptS_dir =  "../SWEmatching/Comp1_OptS_QF_fixed/{}.txt".format(model)

    for base_senstivity in ["Regnet400mf", "Regnet800mf", "Regnet2gf", "Regnet6gf"]:
        if model == base_senstivity: continue
        base_senstivity_dir = "../SWEmatching/Comp1_OptS_QF_fixed/base_senstivity_{}/{}.txt".format(base_senstivity, model)
        res = load_curve(OptS_dir, base_senstivity_dir, JPEG_dir, acclb = bounds[model][1], bppup=bounds[model][0])
        plt.figure()
        plt.plot(res["OPTDG"][0],res["OPTDG"][1], label = "OptS", color='r', linewidth=2.0)
        plt.plot(res["OPTD"][0],res["OPTD"][1], label = "OptS_T", color='g', linewidth=2.0)
        plt.plot(res["JPEG"][0],res["JPEG"][1], label = "JPEG", linestyle='--', color='b', linewidth=2.0)
        plt.xlabel("Rate (bpp)")
        plt.ylabel("Accuracy (%)")
        plt.title('TS:{}_to_{}'.format(base_senstivity, model))
        plt.legend(loc='lower right')
        plt.grid(color = 'grey', linestyle = '--', linewidth=1)
        plt.savefig("./Different_Base_Senstivity/base_senstivity_{}_to_{}.png".format(base_senstivity, model),dpi=1200, bbox_inches='tight')

    