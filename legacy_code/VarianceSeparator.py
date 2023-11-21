import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
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

def structure_diff(isolated_structure):
    valid_regions = np.greater(isolated_structure, 0)
    structure_ave = np.mean(isolated_structure, where=valid_regions)
    difference_view = np.zeros_like(isolated_structure)
    difference_view[valid_regions] = np.abs(isolated_structure[valid_regions] - structure_ave)
    return difference_view


def distance_visual(isolated_structure):
    dist_canvas = np.zeros_like(isolated_structure)
    canvas_coords = np.argwhere(isolated_structure > 0)
    index_form = tuple(canvas_coords.T)
    canvas_distances = cdist(canvas_coords, canvas_coords)
    for i in range(canvas_distances.shape[0]):
        print("Ref coord", canvas_coords[i])
        dist_canvas[index_form] = canvas_distances[i]
        dist_canvas[tuple(canvas_coords[i])] = max(canvas_distances[i])
        io.imshow(dist_canvas)
        plt.show()
        print("Logistic version")
        dist_canvas[index_form] = logistic_distance_weighting(canvas_distances[i], k=0.08, amp=255, offset=0.5, show_pdf=False)
        dist_canvas[tuple(canvas_coords[i])] = dist_canvas.max()
        io.imshow(dist_canvas)
        plt.show()
        print("Logistic + Linear version")
        dist_canvas[index_form] = logistic_distance_weighting(canvas_distances[i], k=0.08, amp=255, offset=0.5, linear_weight=True, show_pdf=False)
        dist_canvas[tuple(canvas_coords[i])] = dist_canvas.max()
        io.imshow(dist_canvas)
        plt.show()


def logistic_distance_weighting(distances, k=1, amp=1, offset=0.5, linear_weight=True, show_pdf=False):
    max_value = np.max(distances)
    logistic_weight = 1/(1+np.exp(-1*k*((max_value - distances) - max_value*offset)))
    if linear_weight:
        logistic_weight = logistic_weight*linear_distance_weighting(distances)
    if show_pdf:
        unique_distances = np.unique(distances)
        pdf = 1 / (1 + np.exp(-1 * k * ((max_value - unique_distances) - max_value * offset)))
        plt.plot(unique_distances, pdf)
        plt.show()
        if linear_weight:
            plt.plot(unique_distances, pdf*linear_distance_weighting(unique_distances))
            plt.show()
    return logistic_weight*amp


def linear_distance_weighting(distances):
    max_value = np.max(distances)
    inversed_weighting = np.abs(distances-max_value)/max_value
    return inversed_weighting


def continuous_spatial_variance(isolated_structure, k=1, offset=0.5, view_pdf=False):
    non_zero_regions = np.greater(isolated_structure, 0)
    region_pixel_count = non_zero_regions.astype('uint8').sum()
    dist_canvas = np.zeros_like(isolated_structure)
    distance_coords = np.argwhere(isolated_structure > 0)
    index_form = tuple(distance_coords.T)
    pixel_distances = cdist(distance_coords, distance_coords)
    max_dist = np.max(pixel_distances)
    weighted_var_store = np.zeros(pixel_distances.shape[0])
    for i in range(pixel_distances.shape[0]):
        dist_canvas[index_form] = logistic_distance_weighting(pixel_distances[i], k=k, amp=max_dist, offset=offset, show_pdf=view_pdf)
        # dist_canvas[index_form] = linear_distance_weighting(pixel_distances[i])
        normalized_dists = dist_canvas/np.max(dist_canvas)
        denominator_weighting = region_pixel_count - np.sum(normalized_dists)
        if denominator_weighting < 0:
            print("Out of bounds", denominator_weighting)
        weighted_intensities = isolated_structure * normalized_dists
        '''io.imshow(normalized_dists)
        plt.show()
        io.imshow(weighted_intensities)
        plt.show()'''
        weighted_var = np.std(weighted_intensities, where=non_zero_regions, ddof=denominator_weighting)
        weighted_var_store[i] = weighted_var
    print(weighted_var_store.shape)
    weighted_var_array = np.zeros_like(isolated_structure)
    weighted_var_array[index_form] = weighted_var_store
    return weighted_var_array




def structure_variance(threshed_image, labelled_array, str_checked, padding=1):
    structure_coords = np.nonzero(labelled_array == str_checked)
    structure_mask = labelled_array == str_checked
    y_min = np.min(structure_coords[0])
    y_max = np.max(structure_coords[0])
    x_min = np.min(structure_coords[1])
    x_max = np.max(structure_coords[1])
    # io.imshow(labeled_im2[y_min:y_max, x_min:x_max]) # indexing example

    boundary_tuples = [(y_min, y_max), (x_min, x_max)]
    structure_dimensions = []
    for bt in boundary_tuples:
        structure_dimensions.append(abs(bt[1] - bt[0] + 2 * padding))
    structure_canvas = np.zeros(tuple(structure_dimensions))
    isolated_structure = threshed_image*structure_mask
    structure_canvas[padding:-1 * padding, padding:-1 * padding] = isolated_structure[y_min:y_max, x_min:x_max]
    universal_variance = continuous_spatial_variance(structure_canvas, k=0.1, offset=0.8, view_pdf=False)
    structure_topography = (universal_variance/np.max(universal_variance))*structure_canvas
    print("Universal variance")
    io.imshow((universal_variance/np.max(universal_variance))*255)
    plt.show()
    io.imshow((structure_topography/np.max(structure_topography))*255)
    plt.show()
    diff_version = structure_diff(structure_canvas)
    io.imshow((structure_canvas/np.max(structure_canvas))*255)
    plt.show()
    variance_mapping1 = variance_windows(diff_version, padding)
    io.imshow((variance_mapping1/np.max(variance_mapping1))*255)
    plt.show()
    structure_canvas[padding:-1*padding, padding:-1*padding] = variance_mapping1
    variance_mapping2 = variance_windows(structure_canvas, padding)
    io.imshow((variance_mapping2/np.max(variance_mapping2))*255)
    plt.show()

def variance_windows(isolated_struct, padding):
    windows_array = view_as_windows(isolated_struct, 1+2*padding, step=1)

    # Attempt at reducing windows processed that are empty windows (background only)
    window_sums = np.sum(np.sum(windows_array, axis=-1), axis=-1)
    non_background_windows = np.nonzero(window_sums)  # determines which windows contain some intensity
    valid_windows = windows_array[non_background_windows]
    centre_check = np.nonzero(valid_windows[..., padding, padding])  # determines which windows contain a value for the centred element
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
structure_variance(threshed_image, labeled_im2, 52, 1)
io.imshow(labeled_im2)
plt.show()
# print("New structure count", label_count2)
'''squared_image = np.square(size_threshed).astype('int32')
neighbourhood = morphology.rectangle(25, 25)'''
# The coordinate acquisition to be performed using tuples and loops based on the length of the tuple where the first element is the min and the last is the max









