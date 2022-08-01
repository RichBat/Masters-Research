import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import data, io
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold
import tifffile


json_path = "C:\\RESEARCH\\Mitophagy_data\\gui params\\rensu_thresholds.json"
orig_im_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"
auto_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Thresholded\\"
save_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MIP\\"

orig_images = [[f, orig_im_path + f] for f in listdir(orig_im_path) if isfile(join(orig_im_path, f)) and f.endswith('.tif')]
thresholded_images = [[f, auto_path + f] for f in listdir(auto_path) if isfile(join(auto_path, f)) and f.endswith('.tif')]
orig_threshold_mapping = {}
for f1 in orig_images:
    for f2 in thresholded_images:
        if f1[0].split('.')[0] == f2[0].split('-')[0]:
            orig_threshold_mapping[f2[0]] = f1[1]
param_file = open(json_path)
json_thresholds = json.load(param_file)
param_file.close()
print("Original Images:")
for oi in orig_images:
    print(oi[0])
    img = io.imread(oi[1]).astype('int')
    low_thresh = json_thresholds[oi[0]]["low"]
    high_thresh = json_thresholds[oi[0]]["high"]
    thresh_mask = apply_hysteresis_threshold(img, low_thresh, high_thresh).astype('int')
    zero_plane = np.zeros_like(img)
    masked = img * thresh_mask
    orig_rgb = np.stack((img, img, img), axis=-1)
    thresh_rgb = np.stack((thresh_mask * 255, zero_plane, zero_plane), axis=-1)
    combined_stack = orig_rgb * 0.65 + thresh_rgb * 0.35
    mip_orig = np.clip(np.amax(combined_stack, axis=0).astype('uint8'), 0, 255)
    io.imsave(save_path + oi[0], mip_orig)
print("Thresh Images:")
for ti in thresholded_images:
    print(ti[0])
    img = io.imread(ti[1]).astype('int')
    orig_img = io.imread(orig_threshold_mapping[ti[0]]).astype('int')
    option_set = ti[0].split('-')
    name = option_set[0]
    thresh_type = option_set[1].split('#')[1]
    thresh_option = option_set[2].split('#')[1].split('.')[0]
    option_label = 0
    if thresh_type == "Inverted":
        option_label += 0
    else:
        option_label += 3
    option_label += int(thresh_option)
    print(ti[0], option_label)
    thresh_mask = (img > 0).astype('int')
    zero_plane = np.zeros_like(img)
    masked = img * thresh_mask
    orig_rgb = np.stack((orig_img, orig_img, orig_img), axis=-1)
    thresh_rgb = np.stack((thresh_mask * 255, zero_plane, zero_plane), axis=-1)
    combined_stack = orig_rgb * 0.65 + thresh_rgb * 0.35
    mip_orig = np.clip(np.amax(combined_stack, axis=0).astype('uint8'), 0, 255)
    io.imsave(save_path + name + "Option" + str(option_label) + ".tif", mip_orig)

