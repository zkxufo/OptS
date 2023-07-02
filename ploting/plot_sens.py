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


plot_data = lambda x: np.load(x)
channel_colors = {"Y": 'k', "Cb" : 'b', "Cr": 'r'}

# for model in ["mobilenet_v2", "Regnet400mf", "Resnet18", "Squeezenet", "Shufflenetv2", "Mnasnet", "Alexnet"]:
#     plt.figure()
#     for channel in ["Y", "Cb", "Cr"]:
#         plt.plot(plot_data("../SWEmatching/SenMap/{}{}.npy".format(channel, model)), channel_colors[channel], label="{} channel".format(channel))    
#     plt.xlabel("DCT Index")
#     plt.legend(loc='upper right')
#     plt.tight_layout() 
#     plt.savefig("./Senstivity/{}.png".format(model),dpi=900)

# plot_data = lambda x: np.load(x) /np.load(x)[0]
# for channel in ["Y", "Cb", "Cr"]:
#     plt.figure()
#     for model in ["Regnet400mf", "Regnet800mf", "Regnet8gf"]:
#         plt.plot(plot_data("../SWEmatching/SenMap_HRS/{}{}.npy".format(channel, model)), channel_colors[channel], label="HRS+{}{} {} channel".format(channel, model, channel))
#         plt.legend(loc='best')
#     plt.savefig("./Senstivity_crossModel/HRS_{}_channel_{}.png".format(channel, model),dpi=900)

#     plt.figure()
#     for model in ["Regnet400mf", "Regnet800mf", "Regnet8gf"]:
#         plt.plot(plot_data("../SWEmatching/SenMap/{}{}.npy".format(channel, model)), channel_colors[channel], label="{}{} {} channel".format(channel,model, channel))
#         plt.legend(loc='best')
#     plt.savefig("./Senstivity_crossModel/{}_channel_{}.png".format(channel, model),dpi=900)


plot_data = lambda x: np.load(x) /np.load(x)[0]
for channel in ["Y", "Cb", "Cr"]:
    plt.figure()
    for model in ["Regnet400mf", "mobilenet_v2"]:
        plt.plot(plot_data("../DCT_coefficient/SenMap_HRS/{}{}.npy".format(channel, model)), label="HRS+{}{} {} channel".format(channel, model, channel))
        plt.legend(loc='best')
    plt.savefig("./Senstivity_crossModel/HRS_{}_channel_{}.png".format(channel, model),dpi=900)

    plt.figure()
    for model in ["Regnet400mf", "mobilenet_v2"]:
        plt.plot(plot_data("../SWEmatching/SenMap/{}{}.npy".format(channel, model)), label="{}{} {} channel".format(channel,model, channel))
        plt.legend(loc='best')
    plt.savefig("./Senstivity_crossModel/{}_channel_{}.png".format(channel, model),dpi=900)