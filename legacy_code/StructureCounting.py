import json
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import segmentation
import skimage
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
import tifffile
from knee_locator import KneeLocator
import time
from scipy import ndimage as ndi
from skimage import measure

# CCCP_1C=1T=0-ThreshType#Logistic-Option#2.tif
file_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Thresholded\\CCCP_1C=1T=0-ThreshType#Logistic-Option#2.tif"
image = io.imread(file_path)
base_image = np.amax(image, axis=0)
binary_mask = (image > 0).astype('int')
binary_mask = np.amax(binary_mask, axis=0)
#base_image = base_image[1180:1380, 681:882]
#binary_mask = binary_mask[1180:1380, 681:882]
'''io.imshow(binary_mask, axis=0)
plt.show()'''
mask_rank = binary_mask.ndim
structure_array = ndi.generate_binary_structure(mask_rank, 1)
#structure_array = ndi.iterate_structure(structure_array, 2)
print(mask_rank)
print(structure_array.ndim)
print(structure_array)

label_array, num_structs = ndi.label(binary_mask, structure_array)
label_details, label_counts = np.unique(label_array, return_counts=True)

#label_array2, num_structs2 = measure.label(binary_mask)
print("Number of Structures", num_structs)
visual_label = (label_array > 0).astype('int')

'''
io.imshow(label_array)
plt.show()'''
distance = ndi.distance_transform_edt(binary_mask)
foot = np.ones((30, 30))
coords = skimage.feature.peak_local_max(distance, footprint=foot, labels=binary_mask)
mask2 = np.zeros(binary_mask.shape, dtype=bool)
mask2[tuple(coords.T)] = True
markers, _ = ndi.label(mask2, structure_array)
print("Structure Count2:", _)
labels2 = segmentation.watershed(-distance, markers, mask=binary_mask)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(base_image)
ax2.imshow(labels2)
plt.show()
''''for z in range(visual_label.shape[0]):
    print("Slice", z+1)
    io.imshow(visual_label[z])
    plt.show()
    '''''''io.imshow(label_array2[z])
    plt.show()'''