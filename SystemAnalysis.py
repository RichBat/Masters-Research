import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
import tifffile
from knee_locator import KneeLocator

from CleanThresholder import AutoThresholder

class thresholding_metrics(AutoThresholder):

    def __init__(self, input_paths, deconv_paths=None):
        self.low_thresholds = {}
        AutoThresholder.__init__(self, input_paths, deconv_paths)

    def _prepare_image(self, image, filename):
        gray_image = self._grayscale(image)
        image_set = self._timeframe_sep(gray_image, filename)
        return image_set

    def analyze_low_thresholds(self, save_path=None):
        for f in self.file_list:
            image = io.imread(f[0])
            time_set = self._prepare_image(image, f[1])
            for t in range(0, len(time_set)):
                img = time_set[t]
                normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2)
                log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2)
                otsu_thresh = threshold_otsu(img)
                valid = True
                if otsu_thresh <= normal_knee:
                    chosen_knee = normal_knee
                else:
                    chosen_knee = log_knee
                if log_knee <= threshold_triangle(img):
                    valid = False
                if save_path is not None:
                    key_string = str(f[1]) + " " + str(t)
                    self.low_thresholds[key_string] = {"Normal": str(normal_knee), "Log": str(log_knee), "Otsu": str(otsu_thresh), "Chosen": str(chosen_knee),
                                                      "Triangle": str(threshold_triangle(img)), "Valid": str(valid)}
                else:
                    self.low_thresholds[(f[1], t)] = {"Normal":normal_knee, "Log":log_knee, "Otsu":otsu_thresh, "Chosen":chosen_knee,
                                                      "Triangle":threshold_triangle(img), "Valid":valid}
        if save_path is not None:
            with open(save_path, 'w') as j:
                json.dump(self.low_thresholds, j)

