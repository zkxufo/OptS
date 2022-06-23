import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
from scipy.stats import bootstrap
import argparse
def main(model = 'alexnet'):
    Y_sen_list = np.load("./grad/Y_sen_list" + model + ".npy")
    Cb_sen_list = np.load("./grad/Cr_sen_list" + model + ".npy")
    Cr_sen_list = np.load("./grad/Cb_sen_list" + model + ".npy")
    zigzag = get_zigzag()
    lst_length = Y_sen_list.shape[0]
    Y_sen_img = np.zeros((64,lst_length))
    Cb_sen_img = np.zeros((64,lst_length))
    Cr_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Y_sen_img[zigzag[i,j]] = Y_sen_list[:,i,j]
    del Y_sen_list
    for i in range(8):
        for j in range(8):
            Cb_sen_img[zigzag[i,j]] = Cb_sen_list[:,i,j]
    del Cb_sen_list
    for i in range(8):
        for j in range(8):
            Cr_sen_img[zigzag[i,j]] = Cr_sen_list[:,i,j]
    del Cr_sen_list


    bottom_lst = []
    top_lst = []
    mean_lst = []
    # figure(figsize=(10, 8), dpi=1024)
    for i in tqdm(range(64)):
        bottom,top = list(bootstrap((Y_sen_img[i],), np.mean, confidence_level=0.95,n_resamples=100).confidence_interval)
        mean = np.mean((bottom,top))
        bottom_lst.append(bottom)
        top_lst.append(top)
        mean_lst.append(mean)
    plt.figure(figsize=(10,8))
    # plt.plot(bottom_lst)
    # plt.plot(top_lst)
    # plt.plot(mean_lst)
    mean_lst = np.array(mean_lst)
    Range = np.max(mean_lst)-np.min(mean_lst)
    mean_lst = mean_lst/Range
    mean_val = np.mean(mean_lst)
    mean_lst = mean_lst-mean_val+1
    #
    plt.plot(mean_lst)
    #
    plt.xticks(np.arange(1,65,4))
    plt.title('Y channel L1 sensitivity, per image')
    plt.savefig("Y"+model+".pdf")
    plt.figure().clear()
    print("y: ",mean_lst)


    bottom_lst = []
    top_lst = []
    mean_lst = []
    for i in tqdm(range(64)):
        bottom,top = list(bootstrap((Cb_sen_img[i],), np.mean, confidence_level=0.95,n_resamples=100).confidence_interval)
        mean = np.mean((bottom,top))
        bottom_lst.append(bottom)
        top_lst.append(top)
        mean_lst.append(mean)
    # plt.plot(bottom_lst)
    # plt.plot(top_lst)
    # plt.plot(mean_lst)
    mean_lst = np.array(mean_lst)
    Range = np.max(mean_lst)-np.min(mean_lst)
    mean_lst = mean_lst/Range
    mean_val = np.mean(mean_lst)
    mean_lst = mean_lst-mean_val+1
    #
    plt.plot(mean_lst)
    #
    plt.xticks(np.arange(1,65,4))
    plt.title('Cb channel L1 sensitivity, per image')
    plt.savefig("Cb"+model+".pdf")
    plt.figure().clear()
    print("Cb: ",mean_lst)


    bottom_lst = []
    top_lst = []
    mean_lst = []
    for i in tqdm(range(64)):
        bottom,top = list(bootstrap((Cr_sen_img[i],), np.mean, confidence_level=0.95,n_resamples=100).confidence_interval)
        mean = np.mean((bottom,top))
        bottom_lst.append(bottom)
        top_lst.append(top)
        mean_lst.append(mean)
    # plt.plot(bottom_lst)
    # plt.plot(top_lst)
    # plt.plot(mean_lst)
    mean_lst = np.array(mean_lst)
    Range = np.max(mean_lst)-np.min(mean_lst)
    mean_lst = mean_lst/Range
    mean_val = np.mean(mean_lst)
    mean_lst = mean_lst-mean_val+1
    #
    plt.plot(mean_lst)
    #
    plt.xticks(np.arange(1,65,4))
    plt.title('Cr channel L1 sensitivity, per image')
    plt.savefig("Cr"+model+".pdf")
    plt.figure().clear()
    
    print("Cr: ",mean_lst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('-model',type=str, default='Alexnet', help='DNN model')
    args = parser.parse_args()
    main(**vars(args))
