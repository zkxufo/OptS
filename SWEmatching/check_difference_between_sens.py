import numpy as np
model = "mobilenet_v2"
x = np.load("./SenMap_/Y{}.npy".format(model)) - np.load("./SenMap/Y{}.npy".format(model))
print(np.sum(x))

x = np.load("./SenMap_/Cr{}.npy".format(model)) - np.load("./SenMap/Cr{}.npy".format(model))
print(np.sum(x))

x = np.load("./SenMap_/Cb{}.npy".format(model)) - np.load("./SenMap/Cb{}.npy".format(model))
print(np.sum(x))
