import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
# from scipy.stats import bootstrap
import argparse

main_dir = "./grad/"
def main(model = 'alexnet'):
    zigzag = get_zigzag()
    
    Y_sen_list = np.load(main_dir+"Y_sen_list" + model + ".npy")
    lst_length = Y_sen_list.shape[0]
    Y_sen_img = np.zeros((64,lst_length))
    
    for i in range(8):
        for j in range(8):
            Y_sen_img[zigzag[i,j]] = Y_sen_list[:,i,j]
    del Y_sen_list
    Y_mean_lst=np.mean(np.abs(Y_sen_img)**2,1)
    np.save("SenMap/Y"+model, Y_mean_lst)
    del Y_sen_img
    
    Cb_sen_list = np.load(main_dir+"Cr_sen_list" + model + ".npy")
    Cb_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cb_sen_img[zigzag[i,j]] = Cb_sen_list[:,i,j]
    del Cb_sen_list
    Cb_mean_lst=np.mean(np.abs(Cb_sen_img)**2,1)
    np.save("SenMap/Cb"+model, Cb_mean_lst)
    del Cb_sen_img

    Cr_sen_list = np.load(main_dir+"Cb_sen_list" + model + ".npy")
    Cr_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Cr_sen_img[zigzag[i,j]] = Cr_sen_list[:,i,j]
    del Cr_sen_list
    Cr_mean_lst=np.mean(np.abs(Cr_sen_img)**2,1)
    np.save("SenMap/Cr"+model, Cr_mean_lst)
    del Cr_sen_img
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('-model',type=str, default='Alexnet', help='DNN model')
    args = parser.parse_args()
    main(**vars(args))
