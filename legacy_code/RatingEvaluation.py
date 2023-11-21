import json
import numpy as np
from skimage.filters import apply_hysteresis_threshold
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists

Automatic_thresholds = "CompareResults2.json"
manual_threshes_ratings = ""
"C:\\RESEARCH\Mitophagy_data\\Time_split\\Output\\"

class RatingEvaluator:

    def __init__(self, image_path, auto_path=None, manual_threshes=None, manual_ratings=None):
        self.images_path = self.images_only(image_path)
        self.auto_path = auto_path
        self.manual_thresh = self.record_files_path(manual_threshes, 'thresholds')
        self.ratings = self.record_files_path(manual_ratings, 'ratings')
        self.auto_results = self.extract_automatic_ratings()

    def images_only(self, im_path):
        only_images = [[im_path + f, f] for f in listdir(im_path) if isfile(join(im_path, f)) and f.endswith('.tif')]
        return only_images

    def separate_manual_feedback(self, manual_path, ratings_suffix='ratings', threshold_suffix='thresholds'):
        ratings_files = []
        thresholds_files = []
        for f in listdir(manual_path):
            if isfile(join(manual_path, f)) and f.endswith('.json'):
                if f.endswith(ratings_suffix + '.json'):
                    ratings_files.append(manual_path + f)
                if f.endswith(threshold_suffix + '.json'):
                    thresholds_files.append(manual_path + f)
        return ratings_files, thresholds_files

    def record_files_path(self, some_path, suffix):
        files_found = [some_path+f for f in listdir(some_path) if isfile(join(some_path, f)) and f.endswith(suffix + '.json')]
        return files_found

    def extract_automatic_ratings(self):
        json_results = None
        with open(self.auto_path, "r") as j:
            results = json.load(j)
            json_results = results
        j.close()
        return json_results

