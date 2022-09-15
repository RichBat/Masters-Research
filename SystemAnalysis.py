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

# Add expert thresholds for reference thus the low threshold methods can be compare based on closeness to the expert low thresholds

from CleanThresholder import AutoThresholder

class thresholding_metrics(AutoThresholder):

    def __init__(self, input_paths, deconv_paths=None, expert_path=None):
        self.low_thresholds = {}
        self.expert_files = self._extract_expert_files(expert_path)
        self.exp_threshes = None
        self.exp_ratings = None
        self._initialise_expert_info()
        if expert_path is not None:
            self._extract_exp_threshes()
        AutoThresholder.__init__(self, input_paths, deconv_paths)

    def _prepare_image(self, image, filename):
        gray_image = self._grayscale(image)
        image_set = self._timeframe_sep(gray_image, filename)
        return image_set

    def _extract_expert_files(self, exp_path):
        if exp_path is not None:
            threshold_files = [join(exp_path, f) for f in listdir(exp_path) if isfile(join(exp_path, f)) and f.endswith("thresholds.json")]
            rating_files = [join(exp_path, f) for f in listdir(exp_path) if isfile(join(exp_path, f)) and f.endswith("ratings.json")]
            return {'thresholds':threshold_files, 'ratings':rating_files}
        else:
            return None

    def _initialise_expert_info(self):
        '''
        This function will assign the samples being viewed to the expert dictionaries and when these expert results are scraped from the expert json files
        then only samples shared between experts and the currently viewed sample set will be viable
        :return:
        '''
        sample_names = list(self.image_paths)
        self.exp_threshes = {}
        self.exp_ratings = {}
        for s in sample_names:
            self.exp_threshes[s] = None
            self.exp_ratings[s] = None


    def _extract_exp_threshes(self):
        '''
        This function will iterate through the expert threshold results and scrape the sample specific results, this results will be stored as a list of tuple
        pairs for (low, high) thresholds. Not all experts evaluated all of the samples thus this check alleviates that.
        :return:
        '''
        expert_threshold_dicts = []  # This will be a compilation of all of the thresholding dictionaries from the different expert files
        for threshold_files in self.expert_files['thresholds']:
            with open(threshold_files, "r") as j:
                expert_threshold_dicts.append(json.load(j))
        for sample_thresholds in list(self.exp_threshes):
            for etd in expert_threshold_dicts:
                if sample_thresholds in etd:
                    if self.exp_threshes[sample_thresholds] is None:
                        self.exp_threshes[sample_thresholds] = []
                    self.exp_threshes[sample_thresholds].append(self._expert_threshold_dict_check(etd[sample_thresholds]))

    def _expert_threshold_dict_check(self, value_dict):
        if len(list(value_dict)) == 2 and "low" in value_dict and "high" in value_dict:
            low_value = value_dict["low"] if type(value_dict["low"]) is int or type(value_dict["low"]) is float else float(value_dict["low"])
            high_value = value_dict["high"] if type(value_dict["high"]) is int or type(value_dict["high"]) is float else float(value_dict["high"])
            thresh_values = (low_value, high_value)
            return thresh_values
        return None

    def _compare_with_experts(self, sample_name, values, choice=0):
        '''
        This method will receive the sample name to select the respective sample from the expert data, determine if it is present, and compare to the auto
        results. The choice parameter is either '0' (both if list/tuple or low thresh), '1' for low thresh and '2' for high thresh. A distance from the expert
        low threshold will be provided. If there are multiple expert values then an average of the distances and the distance from the average expert value
        will be provided. If only one expert value is present then a single value will be returned as a float.
        :param sample_name:
        :param values:
        :param choice:
        :return:
        '''
        if self.exp_threshes[sample_name] is None:
            return None
        else:
            choice_indexes = []

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
            with open(save_path + "lw_thrsh_metrics.json", 'w') as j:
                json.dump(self.low_thresholds, j)

if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    system_analyst = thresholding_metrics(input_path)
    system_analyst.analyze_low_thresholds("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\Low Threshold Metrics\\")

