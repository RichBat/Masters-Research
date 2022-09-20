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
from scipy import ndimage as ndi
import tifffile
from knee_locator import KneeLocator

# Add expert thresholds for reference thus the low threshold methods can be compare based on closeness to the expert low thresholds

from CleanThresholder import AutoThresholder

class thresholding_metrics(AutoThresholder):

    def __init__(self, input_paths, deconv_paths=None, expert_path=None):
        AutoThresholder.__init__(self, input_paths, deconv_paths)
        self.low_thresholds = {}
        self.expert_files = self._extract_expert_files(expert_path)
        self.exp_threshes = None
        self.exp_ratings = None
        self._initialise_expert_info()
        self.expert_path = expert_path
        if expert_path is not None:
            self._extract_exp_threshes()

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
        pairs for (low, high) thresholds. Not all experts evaluated all of the samples thus this check alleviates that. This may need to be adjusted to include
        the ranking since anonymising the expert values could affect the rank pairings unless in the expert ranking this is included.
        :return:
        '''
        expert_threshold_dicts = []  # This will be a compilation of all of the thresholding dictionaries from the different expert files
        for threshold_files in self.expert_files['thresholds']:
            with open(threshold_files, "r") as j:
                expert_threshold_dicts.append(json.load(j))
        for sample_thresholds in list(self.exp_threshes):
            expert_none_count = 0
            for etd in expert_threshold_dicts:
                if self.exp_threshes[sample_thresholds] is None:
                    self.exp_threshes[sample_thresholds] = []
                if sample_thresholds in etd:
                    self.exp_threshes[sample_thresholds].append(self._expert_threshold_dict_check(etd[sample_thresholds]))
                else:
                    self.exp_threshes[sample_thresholds].append(None)  # This will account for some experts not evaluating a sample
                    expert_none_count += 1
            if expert_none_count == len(self.exp_threshes[sample_thresholds]):
                self.exp_threshes[sample_thresholds] = None

    def _expert_threshold_dict_check(self, value_dict):
        '''
        This function scrapes through the expert dictionary provided and searches for the sample. If the sample is present and so are both a low and high
        threshold value then the tuple of the threshold values will be returned. If the sample was not reviewed by the expert then None is returned.
        :param value_dict: The sample dictionary which is a child of the expert dictionary
        :return:
        '''
        if len(list(value_dict)) == 2 and "low" in value_dict and "high" in value_dict:
            low_value = value_dict["low"] if type(value_dict["low"]) is int or type(value_dict["low"]) is float else float(value_dict["low"])
            high_value = value_dict["high"] if type(value_dict["high"]) is int or type(value_dict["high"]) is float else float(value_dict["high"])
            thresh_values = (low_value, high_value)
            return thresh_values
        return None

    def _image_analysis(self, image1, image2):
        '''
        This method will compare the visual differences between the images provided. The images will be the intensity images.
        :param image1:
        :param image2:
        :return:
        '''
        voxel_count_difference = np.count_nonzero(image1) - np.count_nonzero(image2)
        if voxel_count_difference > 0:
            larger_image = 1
        elif voxel_count_difference < 0:
            larger_image = 2
        else:
            larger_image = 0
        voxel_intensity_difference = abs(image1 - image2)

    def _structure_overlap(self, image1, image2):
        '''
        This method is designed to determine what labelled structures from each of the images overlap. From here the percentage overlap relative to the
        overlapping structures complete volumes could be calculated as well as a relationship between structure aggregation? (one large overlaps with many
        small).
        :param image1:
        :param image2:
        :return:
        '''
        overlap_image = np.logical_and(np.greater(image1, 0), np.greater(image2, 0))
        print("Complete overlap")
        self._composite_image_preview(image1, image2, overlap_image)
        overlap_regions, overlap_count = ndi.labels(overlap_image)
        structure_seg1, structure_count1 = ndi.labels(image1)
        structure_seg2, structure_count2 = ndi.labels(image2)
        for over_regions in range(1, overlap_count):
            isolated_overlap = np.where(overlap_regions == over_regions)
            image1_overlap = structure_seg1 * isolated_overlap
            image2_overlap = structure_seg2 * isolated_overlap
            image1_labels = np.unique(image1_overlap).tolist()
            image1_labels.remove(0)
            image2_labels = np.unique(image2_overlap).tolist()
            image2_labels.remove(0)
            ''' This will provide the labels for the structures that fall within the overlapping zones. Relationships within the overlap zone should be 1 to 1
            or 1 to many structures for an overlapping region. The reference image (image1) could have the same structure fall into multiple overlapping zones
            so these respective overlaps should be aggregated prior to measurements being performed.
            '''

    def _composite_image_preview(self, image1, image2, overlap_region):
        '''
        This method will compose an image between the two 3D images (which will be an MIP for visualisation) with the overlap region of interest displayed.
        The images must be 3D or 2D and grayscale
        :param image1:
        :param image2:
        :param overlap_region:
        :return:
        '''
        if image1.shape != image2.shape:
            raise Exception("Image shapes do not match")
        image_shape = list(image1.shape)
        if len(image_shape) > 2:
            image1 = np.amax(image1, axis=0)
            image2 = np.amax(image2, axis=0)
        if len(overlap_region.shape) > 2:
            overlap_region = np.amax(overlap_region, axis=0)
        image_shape.append(3)
        composite_image = np.zeros(tuple(image_shape))
        composite_image[..., 0] = image1
        composite_image[..., 1] = image2
        composite_image[..., 2] = overlap_region
        io.imshow(composite_image)
        plt.show()



    def _compare_with_experts(self, sample_name, values, choice=2):
        '''
        This method will receive the sample name to select the respective sample from the expert data, determine if it is present, and compare to the auto
        results. The choice parameter is either '2' (both if list/tuple or low thresh), '0' for low thresh and '1' for high thresh. A distance from the expert
        low threshold will be provided. If there are multiple expert values then an average of the distances and the distance from the average expert value
        will be provided. If only one expert value is present then a single value will be returned as a float.

        This might be adjusted in future to take expert ranking into account.
        :param sample_name:
        :param values:
        :param choice:
        :return:
        '''
        if self.exp_threshes[sample_name] is None:
            return None
        else:
            expert_values = self.exp_threshes[sample_name]
            comparison_results = []
            if choice != 2 or len(values) == 1:
                provided_value = values if (type(values) is not tuple or type(values) is not list) else values[choice]
                if choice == 2:
                    # defaults choice to 0 if only one value (low or high) has been provided thus low is assumed
                    choice = 0
                ev_collection = 0
                ev_distances = 0
                not_none_ev = 0
                for ev in expert_values:
                    if ev is not None:
                        comparison_results.append(provided_value-ev[choice])
                        ev_collection += ev[choice]
                        ev_distances += provided_value-ev[choice]
                        not_none_ev += 1
                    else:
                        comparison_results.append(None)
                if len(comparison_results) > 1:
                    average_expert = (provided_value - (ev_collection/not_none_ev))
                    average_distance = ev_distances/not_none_ev
                    comparison_results.append('mean')
                    comparison_results.append([average_expert, average_distance])
                return comparison_results
            else:
                provided_value = values
                ev_collection = [0, 0]
                ev_distances = [0, 0]
                not_none_ev = 0
                for ev in expert_values:
                    if ev is not None:
                        comparison_results.append((provided_value[0]-ev[0], provided_value[1]-ev[1]))
                        ev_collection[0] += ev[0]
                        ev_collection[1] += ev[1]
                        ev_distances[0] += provided_value[0] - ev[0]
                        ev_distances[1] += provided_value[1] - ev[1]
                        not_none_ev += 1
                    else:
                        comparison_results.append(None)
                if len(expert_values) > 1:
                    comparison_results.append('mean')
                    comparison_results.append([(ev_collection[0]/not_none_ev, ev_collection[1]/not_none_ev),
                                               (ev_distances[0]/not_none_ev, ev_distances[1]/not_none_ev)])
                return comparison_results

    def _expert_ranking_weight(self, ranking, sample_name, expert_index):
        '''
        This function will be used to weight the deviation between the expert result and the automated result if the ranking is present. Since the thresholds
        are anonymised between experts for threshold aggregation the associated rankings need to be tracked. Either the threshold extraction needs to
        correctly number the associated ranking or include it in the self.exp_threshes that will be a tuple of (low, high, rank) where a value of None can be
        provided if there is no rank. The method used to weight the rankings will be based on the numeric weighting where 3 is the centre value and a rank
        lower than 3 (<3) will be worse .
        :param ranking:
        :param sample_name:
        :param expert_index:
        :return:
        '''
        print("none")

    def _low_thresh_compare(self, value_sequence, sample_name):
        '''
        This will iterate through the provided low thresholds (knee types, Otsu, etc.) and evaluate the distance from the experts. This will be used to feed
        an organized dictionary for all samples to be aggregated.
        :param value_sequence: dict of values with measurement type as key
        :return:
        '''
        distance_results = {}
        total_expert_count = 1
        for thresh_type, thresh_value in value_sequence.items():
            expert_count = 1  # this will be used to track the number of experts for a sample
            distances = self._compare_with_experts(sample_name, thresh_value, choice=0)  # can be none for expert missing (None, 5, 6, None, [averages])
            mean_check = False
            ''' distances will be a list of values which can contain None values. distances is a list and contains a None value then the expert count must 
            increment
            '''
            if "Sample" not in distance_results:
                distance_results["Sample"] = []
            if "Thresh" not in distance_results:
                distance_results["Thresh"] = []
            distance_results["Sample"].append(sample_name)
            distance_results["Thresh"].append(thresh_type)

            if type(distances) is list and distances is not None:
                for dist in distances:
                    if not mean_check and dist != "mean":
                        expert_name = "Exp" + str(expert_count)
                        if expert_name not in distance_results:
                            distance_results[expert_name] = []
                        distance_results[expert_name].append(dist)  # this could be a None value but is not an issue, expected as it will be evaluated later
                        expert_count += 1
                    elif mean_check and dist != "mean":
                        if "MeanExp" not in distance_results:
                            distance_results["MeanExp"] = []
                            distance_results["ExpMean"] = []
                        distance_results["MeanExp"].append(dist[0])
                        distance_results["ExpMean"].append(dist[1])
                    else:  # this check will denote that the next entry is a 2 item list for the averages across the experts
                        mean_check = True
            else:
                distance_results["Exp1"] = None
                distance_results["MeanExp"] = None
                distance_results["ExpMean"] = None
            total_expert_count = max(total_expert_count, expert_count)  # remain unchanged after 1st thresh_type since expert count independent of thresh_type

        return distance_results, total_expert_count

    def _aggregate_across_samples(self, sample_values, evaluation_type=None):
        '''
        This method will aggregate all of the values by sample, the evaluation_type will determine the analysis for the per sample values (defaulted to None)
        and organise these values for a pandas table which can be manipulated and output to csv.
        :param sample_values:
        :param evaluation_type:
        :return:
        '''
        total_expert_count = 1  # this will monitor the number of experts to retroactively place None values for missing evaluations
        aggregate_dict = {}
        if evaluation_type is None:
            for sample_name, sample_results in sample_values.items():
                compared_values, expert_count = self._low_thresh_compare(sample_results, sample_name)  # make sure "Valid" will not be in the sample_results
                number_of_entries = len(compared_values["Sample"])
                for noe in range(number_of_entries):
                    for cv in list(compared_values):  # this will initialise the aggregate dictionary
                        if cv not in aggregate_dict:
                            aggregate_dict[cv] = []
                        aggregate_dict[cv].append(compared_values[cv][noe])

        pandas_view = pd.DataFrame.from_dict(aggregate_dict)
        print(pandas_view)

    def analyze_low_thresholds(self, save_path=None, experts=True):
        values_for_experts = {}
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
                if self.expert_path is not None:
                    values_for_experts[f[1]] = {"Normal": normal_knee, "Log": log_knee, "Otsu": otsu_thresh,
                                                      "Triangle": threshold_triangle(img)}
        if experts and self.expert_path is not None:
            self._aggregate_across_samples(values_for_experts)
        if save_path is not None:
            with open(save_path + "lw_thrsh_metrics.json", 'w') as j:
                json.dump(self.low_thresholds, j)

if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    system_analyst = thresholding_metrics(input_path, expert_path="C:\\RESEARCH\\Mitophagy_data\\gui params\\")
    system_analyst.analyze_low_thresholds("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\Low Threshold Metrics\\")

