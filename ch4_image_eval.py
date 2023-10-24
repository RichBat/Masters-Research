import numpy as np
import math

import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_sauvola, threshold_otsu, threshold_li
from skimage.filters.rank import otsu
from skimage import morphology
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, variation_of_information
import warnings
import json
import seaborn as sns

from CleanThresholder import AutoThresholder


def overlay_raw(raw_path, hyst_addition, adpt_addition, save_addition):
    raw_files = [f for f in listdir(raw_path) if isfile(join(raw_path, f)) and f.endswith('.tif')]

    for rf in raw_files:
        raw_im = io.imread(join(raw_path, rf))
        hyst_im = np.clip(io.imread(join(join(raw_path, hyst_addition), "Hyst_" + rf)), 0, 1)*raw_im
        adapt_im = np.clip(io.imread(join(join(raw_path, adpt_addition), "Adpt_" + rf)), 0, 1)*raw_im

        raw_im = np.amax(raw_im, axis=0)
        hyst_im = np.amax(hyst_im, axis=0)
        adapt_im = np.amax(adapt_im, axis=0)

        overlay = np.stack([hyst_im, adapt_im, raw_im], axis=-1)
        save_name = rf.split('.')[0] + "_Overlay.png"
        io.imsave(join(join(raw_path, save_addition), save_name), overlay)

def thresh_overlay(raw_path, hyst_addition, adpt_addition, save_addition):
    raw_files = [f for f in listdir(raw_path) if isfile(join(raw_path, f)) and f.endswith('.tif')]
    ihh_auto = AutoThresholder([raw_path])

    def overlay(hyst_im, adapt_im, thresh_im, save_name):
        flattened_im = np.stack([np.amax(thresh_im, axis=0), np.amax(hyst_im, axis=0), np.amax(adapt_im, axis=0)], axis=-1)
        io.imsave(join(join(raw_path, save_addition), save_name), flattened_im)

    for rf in raw_files:
        sample_name = rf.split('.')[0]
        raw_im = io.imread(join(raw_path, rf))
        hyst_im = io.imread(join(join(raw_path, hyst_addition), "Hyst_" + rf))
        adapt_im = io.imread(join(join(raw_path, adpt_addition), "Adpt_" + rf))
        otsu_thresh = threshold_otsu(raw_im)
        thresh_im = (raw_im > otsu_thresh).astype('uint8')*255
        overlay(hyst_im, adapt_im, thresh_im, sample_name + "_Otsu.png")
        li_thresh = threshold_li(raw_im)
        thresh_im = (raw_im > li_thresh).astype('uint8') * 255
        overlay(hyst_im, adapt_im, thresh_im, sample_name + "_Li.png")
        for n in [75, 101, 151]:
            window = morphology.cube(n)[0:1]
            l_otsu_thresh = otsu(raw_im, window)
            thresh_im = (raw_im > l_otsu_thresh).astype('uint8') * 255
            overlay(hyst_im, adapt_im, thresh_im, sample_name + "_LOtsu" + str(n) + ".png")
            sauvola_thresh = threshold_sauvola(raw_im, n)
            thresh_im = (raw_im > sauvola_thresh).astype('uint8') * 255
            overlay(hyst_im, adapt_im, thresh_im, sample_name + "_Sauvola" + str(n) + ".png")

        for v in [False, True]:
            for w in [0, 1, 2]:
                high_thresh, low_thresh = ihh_auto.inverted_thresholding_final(raw_im, v, w)
                thresh_im = apply_hysteresis_threshold(raw_im, low_thresh, high_thresh).astype('uint8')*255
                name_variants = sample_name + "_IHH_Voxel" + str(v) + "_Win_" + str(w) + ".png"
                overlay(hyst_im, adapt_im, thresh_im, name_variants)

def li_testing(im_path):
    test_image = io.imread(im_path)
    initial_thresh = threshold_li(test_image)
    guess_thresh = threshold_li(test_image, initial_guess=test_image.min()+1)
    print("initial threshold", initial_thresh)
    print("thresh with guess", guess_thresh)
    plt.figure()
    output_im = (test_image > initial_thresh).astype('uint8')
    io.imshow(np.amax(output_im, axis=0))
    plt.figure()
    output_im2 = (test_image > guess_thresh).astype('uint8')
    io.imshow(np.amax(output_im2, axis=0))
    plt.show()

def li_overlay(base_path):
    li_path = join(base_path, "fiji_li")
    li_ims = [li for li in listdir(li_path)]
    for li in li_ims:
        li_im = np.amax(io.imread(join(li_path, li)), axis=0).astype('uint8')
        hyst_im = np.amax(io.imread(join(join(base_path, "hysteresis_output"), "Hyst_" + li)), axis=0).astype('uint8')
        adapt_im = np.amax(io.imread(join(join(base_path, "adaptive_output"), "Adpt_" + li)), axis=0).astype('uint8')

        rgb_overlay = np.stack([li_im, hyst_im, adapt_im], axis=-1)
        li_name = li.split('.')[0] + '_Li.png'
        io.imsave(join(join(base_path, "li_overlays2"), li_name), rgb_overlay)

if __name__ == "__main__":
    base_path = "F:\\clean images\\real_samples\\"
    li_overlay(base_path)
    #li_testing(join(base_path, "LML_3C=2.tif"))
    #thresh_overlay(base_path, "hysteresis_output", "adaptive_output", "Thresh_Overlays")
    #overlay_raw(base_path, "hysteresis_output", "adaptive_output", "Raw_Overlay")