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
test_image_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MIP\\CCCP_2C=0T=0Option0.tif"

test_image = io.imread(test_image_path)
"""
Below is a roughshod approach to extract the threshold mask without investigating the original threshold values
"""
#test_array = np.array([[2, 2, 2], [2, 0, 2], [6, 4, 1]])

def structure_variance(labelled_array, str_checked):
    structure_coords = np.nonzero(labelled_array == str_checked)
    y_min = np.min(structure_coords[0])
    y_max = np.max(structure_coords[0])
    x_min = np.min(structure_coords[1])
    x_max = np.max(structure_coords[1])
    # io.imshow(labeled_im2[y_min:y_max, x_min:x_max]) # indexing example

    PADDING = 1

    boundary_tuples = [(y_min, y_max), (x_min, x_max)]
    structure_dimensions = []
    for bt in boundary_tuples:
        structure_dimensions.append(abs(bt[1] - bt[0] + 2 * PADDING))
    structure_canvas = np.zeros(tuple(structure_dimensions))
    io.imshow(threshed_image[y_min:y_max, x_min:x_max])
    plt.show()
    structure_canvas[1:-1, 1:-1] = threshed_image[y_min:y_max, x_min:x_max]
    variance_mapping1 = variance_windows(structure_canvas)
    io.imshow((variance_mapping1/np.max(variance_mapping1))*255)
    plt.show()
    structure_canvas[1:-1, 1:-1] = variance_mapping1
    variance_mapping2 = variance_windows(structure_canvas)
    io.imshow((variance_mapping2/np.max(variance_mapping2))*255)
    plt.show()

def variance_windows(isolated_struct):
    windows_array = view_as_windows(isolated_struct, 3, step=1)

    # Attempt at reducing windows processed that are empty windows (background only)
    window_sums = np.sum(np.sum(windows_array, axis=-1), axis=-1)
    non_background_windows = np.nonzero(window_sums)  # determines which windows contain some intensity
    valid_windows = windows_array[non_background_windows]
    centre_check = np.nonzero(valid_windows[..., 1, 1])  # determines which windows contain a value for the centred element
    valid_windows = valid_windows[centre_check]  # further filters the number of windows to only the relevant ones
    zero_check = np.greater(valid_windows, 0)  # this will create a mask as zero regions are non-structure and only structure specific variance is desired
    window_variances = np.var(valid_windows, axis=(-1, -2), where=zero_check)  # local variance is calculated and the flattened array is output
    window_mapping = np.zeros(windows_array.shape[0:-2])
    centre_mapping = window_mapping[non_background_windows]
    centre_mapping[centre_check] = window_variances
    window_mapping[non_background_windows] = centre_mapping
    return window_mapping

blank_template = np.zeros_like(test_image) + 1
test_averages = np.stack((test_image.mean(axis=-1),)*3, axis=-1)
average_match = np.equal(test_image, test_averages)
reduced_match = np.invert(np.all(average_match, axis=-1))
threshed_image = test_image.min(axis=-1)*reduced_match

sequential_values = np.arange(0, 81)
test_array = np.reshape(sequential_values, (9, 9))
# print(test_array)
# Apply_parallel attempt (fail)
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
# View_windows attempt (promising)
'''windows_array = view_as_windows(test_array, 2, step=1)
print(windows_array.shape)
print(windows_array)'''

labeled_im, label_count = ndi.label(threshed_image)
# print("Number of structures", label_count)
structure_sufficient = []
retained_structures = np.zeros_like(labeled_im)
for i in range(1, label_count):
    isolated_label = np.equal(labeled_im, i).astype('int')
    if isolated_label.sum() > 70:
        retained_structures = retained_structures + isolated_label

size_threshed = (threshed_image*retained_structures).astype('int32')
labeled_im2, label_count2 = ndi.label(size_threshed)
structure_variance(labeled_im2, 384)
io.imshow(labeled_im2)
plt.show()
# print("New structure count", label_count2)
'''squared_image = np.square(size_threshed).astype('int32')
neighbourhood = morphology.rectangle(25, 25)'''
# The coordinate acquisition to be performed using tuples and loops based on the length of the tuple where the first element is the min and the last is the max









