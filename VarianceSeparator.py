import numpy as np
from scipy import ndimage as ndi
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import rank
from skimage import morphology
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import apply_parallel, view_as_windows
import math
# https://stackoverflow.com/questions/66975708/sliding-window-on-an-image-to-calculate-variance-of-pixels-in-that-window

'''
scipy.ndimage.variance can be used to calculate the variance for specific labels only. Can these labels overlap to be used as neighbourhoods? Perhaps spacing 
the labelled regions (neighbourhoods) such that they don't overlap and are flush with each other and then multiple 3D layers can be made where other layers
contain the overlapped regions such that in 3D they do not overlap but if projected then they will look as desired. These 3D slice can then be run sequentially 
for the variance calculation. The key part would be the time saved by the variance calculation since capturing the neighbourhoods will still be painful.

skimage.util.view_as_windows can get overlapping windows of the region but can consume memory rapidly since a 3x3 window across a 9x9 image will generate a 49
deep array of 3x3 arrays. For my large images it is important to segregate the area of the structures. The window size will improve this efficiency somewhat 
where the window size will somewhat effect the number of strides that can fit.

skimage.util.apply_parallel
'''
test_image_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MIP\\CCCP_1C=1T=0.tif"

test_image = io.imread(test_image_path)
"""
Below is a roughshod approach to extract the threshold mask without investigating the original threshold values
"""
#test_array = np.array([[2, 2, 2], [2, 0, 2], [6, 4, 1]])

blank_template = np.zeros_like(test_image) + 1
test_averages = np.stack((test_image.mean(axis=-1),)*3, axis=-1)
average_match = np.equal(test_image, test_averages)
reduced_match = np.invert(np.all(average_match, axis=-1))
threshed_image = test_image.min(axis=-1)*reduced_match

sequential_values = np.arange(0, 81)
test_array = np.reshape(sequential_values, (9, 9))
print(test_array)
'''array_list = []
def sum_contents(arr):
    global array_list
    if arr.shape == (3, 3):
        array_list.append(arr)
    return np.max(arr)
chunked_array = apply_parallel(sum_contents, test_array, chunks=1, depth=1)
print(type(chunked_array))
print(chunked_array)
for i in array_list:
    print(i)
print(len(array_list))'''
# View_windows attempt
windows_array = view_as_windows(test_array, 2, step=1)
print(windows_array.shape)
print(windows_array)

'''labeled_im, label_count = ndi.label(threshed_image)
print("Number of structures", label_count)
structure_sufficient = []
retained_structures = np.zeros_like(labeled_im)
for i in range(1, label_count):
    isolated_label = np.equal(labeled_im, i).astype('int')
    if isolated_label.sum() > 70:
        retained_structures = retained_structures + isolated_label

size_threshed = (threshed_image*retained_structures).astype('int32')
labeled_im2, label_count2 = ndi.label(size_threshed)
print("New structure count", label_count2)
squared_image = np.square(size_threshed).astype('int32')
neighbourhood = morphology.rectangle(25, 25)
for l in range(1, label_count2):
    isolated_structure = np.equal(labeled_im2, i).astype('int32')
    structure_mean = rank.mean(size_threshed*isolated_structure, selem=neighbourhood)
    squared_structure_mean = rank.mean(squared_image*isolated_structure, selem=neighbourhood)
    structure_mean_sq = np.power(structure_mean, 2)
    neighbour_variance = np.add(squared_structure_mean, structure_mean_sq*-1)
    neighbour_std = np.sqrt(neighbour_variance).clip(0, 255).astype('uint8')
    io.imshow(neighbour_std)
    plt.show()'''







