import json
import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian, threshold_minimum
import tifffile
from knee_locator import KneeLocator
import time
from scipy import ndimage as ndi
import seaborn as sns

def hysteresis_stages(image_path, low_threshold, high_threshold):

    image = io.imread(image_path)
    mip = np.amax(image, axis=0)
    plt.imshow(mip, cmap='viridis')
    plt.axis('off')
    plt.show()

    coloured_regions = np.zeros_like(mip)
    coloured_regions[np.less(mip, low_threshold)] = 85
    coloured_regions[np.less(mip, high_threshold)*np.greater(mip, low_threshold)] = 85*2
    coloured_regions[np.greater_equal(mip, high_threshold)] = 85*3
    plt.imshow(coloured_regions)
    plt.axis('off')
    plt.show()

    threshold_result = apply_hysteresis_threshold(mip, low_threshold, high_threshold).astype(int)
    plt.imshow(threshold_result)
    plt.axis('off')
    plt.show()

def label_structs(image_path, low_thresh, high_thresh):

    image = io.imread(image_path)
    mip = np.amax(image, axis=0)
    mask_low = mip > low_thresh
    labels_low, num_labels = ndi.label(mask_low)
    plt.imshow(labels_low)
    plt.show()
    valid_structures = np.stack([labels_low, mip * (mask_low.astype('int'))],
                                axis=-1)  # The two labels have been stacked
    valid_structures = np.reshape(valid_structures, (
    -1, valid_structures.shape[-1]))  # The dual label image has been flattened save for the label pairs
    # valid_structures = valid_structures[np.nonzero(valid_structures[:, 0])] # The zero label structs have been removed
    sort_indices = np.argsort(valid_structures[:, 0])
    valid_structures = valid_structures[sort_indices]
    label_set, start_index, label_count = np.unique(valid_structures[:, 0], return_index=True, return_counts=True)
    end_index = start_index + label_count
    max_labels = np.zeros(tuple([len(label_set), 2]))
    print(label_set)
    im_canvas = numpy.zeros_like(mip)
    for t in range(len(label_set)):
        max_labels[t, 0] = label_set[t]
        max_labels[t, 1] = valid_structures[slice(start_index[t], end_index[t]), 1].max()
        im_canvas[np.equal(labels_low, t)] = max_labels[t, 1]

    plt.imshow(im_canvas)
    plt.axis('off')
    plt.colorbar
    plt.show()


    '''threshold_result = apply_hysteresis_threshold(mip, low_thresh, high_thresh).astype(int)
    structs, number_of_labels = ndi.label(threshold_result)'''




if __name__ == "__main__":
    input_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"
    cell_name = "CCCP_1C=1T=0.tif"
    label_structs(input_path+cell_name, 15, 180)