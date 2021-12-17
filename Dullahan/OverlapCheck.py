import numpy as np
import copy
import os
import json
import math
from skimage import io

ch1filepath = "N1CCCP_2C=0.tif"
ch2filepath = "N1CCCP_2C=1.tif"
path = "C:\\Users\\richy\\Desktop\\Deconvolved Samples\\"
SavePath = "C:\\Users\\richy\\Desktop\\ColocFiles\\"
Thr1 = 7
Thr2 = 22
ch1Stack = io.imread(path + ch1filepath)
ch2Stack = io.imread(path + ch2filepath)
newShape = list(ch1Stack.shape)
newShape.append(3)
newShape = tuple(newShape)
print(newShape)
combined = np.zeros(shape = newShape)
print(combined.shape)
for z in range(newShape[0]):
    for w in range(newShape[1]):
        for h in range(newShape[2]):
            if(ch1Stack[z,w,h] > Thr1 and ch2Stack[z,w,h] > Thr2):
                combined[z,w,h] = [255,255,255]
            else:
                if ch1Stack[z,w,h] > Thr1:
                    combined[z, w, h, 1] = ch1Stack[z,w,h]
                if ch2Stack[z,w,h] > Thr2:
                    combined[z, w, h, 0] = ch2Stack[z,w,h]
io.imsave(SavePath + "overlap th1_" + str(Thr1) + "th2_" + str(Thr2) + ".tif", combined)