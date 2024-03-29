import json
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
from scipy import ndimage as ndi
import seaborn as sns
import tifffile
from knee_locator import KneeLocator

# Add expert thresholds for reference thus the low threshold methods can be compare based on closeness to the expert low thresholds

from CleanThresholder import AutoThresholder

class thresholding_metrics(AutoThresholder):

    def __init__(self, input_paths, deconv_paths=None, expert_path=None, auto_path=None):
        AutoThresholder.__init__(self, input_paths, deconv_paths)
        self.low_thresholds = {}
        self.expert_files = self._extract_expert_files(expert_path)
        self.exp_threshes = None
        self.exp_ratings = None
        self._initialise_expert_info()
        self.expert_path = expert_path
        self.automatic_thresholds = None
        if auto_path is not None:
            self._extract_stored_automatic(auto_path)
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

    def _extract_stored_automatic(self, auto_path):
        with open(auto_path, "r") as j:
            auto_values = json.load(j)
        thresholds = {}
        for sample, _timeframes in auto_values.items():
            if sample not in thresholds:
                thresholds[sample] = []
            if len(_timeframes.keys()) == 1 and list(_timeframes)[0] == "0":
                thresh = _timeframes["0"]
                thresholds[sample].append(float(thresh["Low"]))
                inverted_thresh = thresh["High"]["Inverted"]
                logistic_thresh = thresh["High"]["Logistic"]
                thresholds[sample].append([[float(inverted_thresh["0"]), float(inverted_thresh["1"]), float(inverted_thresh["2"])],
                                           [float(logistic_thresh["0"]), float(logistic_thresh["1"]), float(logistic_thresh["2"])]])
        self.automatic_thresholds = thresholds

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

    def _image_differences_test(self, image, sample_name):
        expert_threshold = self.exp_threshes[sample_name][0]
        expert_image = self._threshold_image(image, expert_threshold[0], expert_threshold[1])*image
        compare_threshes = []
        compare_threshes = [tuple([expert_threshold[0]-5, expert_threshold[1]+10]), tuple([expert_threshold[0]+10, expert_threshold[1]+10])]
        for t in compare_threshes:
            compare_image = self._threshold_image(image, t[0], t[1])*image
            self._image_analysis(compare_image, expert_image)

    def _image_diff_measure(self, im1, im2):
        binary1 = np.greater(im1, 0).astype(int)
        binary2 = np.greater(im2, 0).astype(int)
        diff_image = binary1 - binary2
        complete_match_to_mismatch = (binary1 * binary2) + diff_image
        diff_ratio = abs(complete_match_to_mismatch.sum())/binary2.sum()
        '''fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(binary1 * binary2, cmap='viridis')
        ax1.set_title("Union of Im1 and Im2 with values [0, 1]")
        ax2.imshow(diff_image, cmap='viridis')
        ax2.set_title("Mismatch of Im1 & Im2 with values [-1, 0]")
        ax3.imshow(complete_match_to_mismatch, cmap='viridis')
        ax3.set_title("Combination of the Union and Mismatch. Values in [-1, 0, 1]")
        plt.show()'''
        diff_vol = np.mean(complete_match_to_mismatch, where=np.abs(complete_match_to_mismatch) > 0)
        return diff_ratio, diff_vol

    def large_excluded_test(self):
        image_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MAX_CCCP_1C=0T=0.tif"
        subject_thresholds = [20, 23]
        target_thresholds = [26, 95]

        puncta_image = io.imread(image_path)
        subject_image = self._threshold_image(puncta_image, subject_thresholds[0], subject_thresholds[1])
        target_image = self._threshold_image(puncta_image, target_thresholds[0], target_thresholds[1])
        io.imshow(np.stack([subject_image*255, target_image*255, np.zeros_like(subject_image)], axis=-1))
        plt.show()
        target_labels, label_count = ndi.label(target_image)
        subject_labels, subject_count = ndi.label(subject_image)
        overlap, penalty = self._distance_from_target(subject_labels, target_labels)
        print(overlap, penalty)

    def _distance_from_target(self, changed_labels, target_labels, thresholds=None):
        over_ratio1, vol_ratio1, structure_pairing1, excluded1 = self._structure_overlap_effi(changed_labels, target_labels,
                                                                                         labels_provided=True)
        # diff_ratio, diff_vol = self._image_diff_measure(target_labels, changed_labels)
        overlap_ratio = np.mean(over_ratio1.sum(axis=0)[1:])
        if overlap_ratio == 0:
            print("THe label image is of shape", changed_labels.shape, target_labels.shape)
            io.imshow(np.greater(np.mean(np.stack([np.greater(changed_labels, 0).astype('uint8'), np.greater(target_labels, 0).astype('uint8'),
                                np.zeros_like(changed_labels).astype('uint8')], axis=-1), axis=0), 0).astype('uint8') * 255)
            plt.show()
        change_vol = np.sum(np.greater(changed_labels, 0).astype(int))
        print("Calculating exclusions")
        def exclusion_ratios(struct_labels, source_im, ref_im):
            ref_label_range, ref_struct_vols = np.unique(ref_im, return_counts=True)
            src_label_range, src_struct_vols = np.unique(source_im, return_counts=True)
            excl_volumes = src_struct_vols[struct_labels]
            src_no_excl = np.delete(src_label_range, struct_labels)
            src_excl_vol = src_struct_vols[src_no_excl]  # x of z
            src_excl_vol = src_excl_vol[src_no_excl > 0]
            ref_label_range, src_label_range = ref_label_range > 0, src_label_range > 0
            ref_struct_vols, src_struct_vols = ref_struct_vols[ref_label_range], src_struct_vols[
                src_label_range]  # ref and source volumes
            src_to_overlap = np.sqrt(np.matmul(over_ratio1[1:, 1:], ref_struct_vols) / src_struct_vols)
            mean_src_overlap = np.mean(src_to_overlap)
            excluded_to_ref = np.mean(excl_volumes) / np.mean(ref_struct_vols)
            print("Excluded struct size to ref size", mean_src_overlap)
            print("Mean ratio excl to ref", excluded_to_ref)
            print("Excluded to src mean", np.mean(ref_struct_vols)/np.mean(src_struct_vols))
            print("Absolute mean diff", abs(np.mean(excl_volumes) - np.mean(ref_struct_vols)))
            print("Absolute mean diff src", abs(np.mean(excl_volumes) - np.mean(src_excl_vol)))
            mean_ref_excl_ratio = 0.5 * excluded_to_ref if excluded_to_ref <= 1 else 0.5 + 0.5 * excluded_to_ref * (np.mean(ref_struct_vols)/np.mean(src_struct_vols))
            remaining_ratio = 1 - mean_ref_excl_ratio
            src_over_detraction = (1 - mean_src_overlap)*remaining_ratio
            weighted_compensation = (np.mean(over_ratio1.sum(axis=0)[1:]) * src_over_detraction + np.mean(
                over_ratio1.sum(axis=0)[1:]) * mean_ref_excl_ratio) * ((excluded1[1][1:].sum()) / change_vol)
            print("From number of excluded by ratio", src_over_detraction)
            print("Excluded relative to ref", mean_ref_excl_ratio)
            print("With weighting", np.mean(over_ratio1.sum(axis=0)[1:]) * src_over_detraction,
                  np.mean(over_ratio1.sum(axis=0)[1:]) * mean_ref_excl_ratio,
                  np.mean(over_ratio1.sum(axis=0)[1:]) * src_over_detraction + np.mean(
                      over_ratio1.sum(axis=0)[1:]) * mean_ref_excl_ratio)
            print("From overlap ratio", np.mean(over_ratio1.sum(axis=0)[1:]) - np.mean(over_ratio1.sum(axis=0)[1:]) * (
                        1 - excluded_to_ref - mean_src_overlap) / 1)
            print("Weighted compensation reduction", np.mean(over_ratio1.sum(axis=0)[1:]) - weighted_compensation)
            return weighted_compensation

        # print("#*#", excluded1, "*", excluded2)
        exclusion_penalty = 0
        if len(excluded1[0]) > 1:
            print("Thresholds1:", excluded1[0])
            # print(excluded1[0][1:], excluded1[1][1:])
            targ_vol = np.sum(np.greater(target_labels, 0).astype(int))
            print("Target volume total", targ_vol)
            print("Change volume total", change_vol)
            print("Change without excl ratio", (change_vol - excluded1[1][1:].sum()) / change_vol)
            exclusion_penalty = exclusion_ratios(excluded1[0][1:], changed_labels, target_labels)
            print("Overlap ratio", np.mean(over_ratio1.sum(axis=0)[1:]))
            print("Penalty", exclusion_penalty)
            excluded_structs = np.zeros_like(changed_labels)
            for t in excluded1[0][1:]:
                excluded_structs += np.equal(changed_labels, t).astype(int)

        return overlap_ratio, exclusion_penalty


    def distribution_from_target(self):
        '''
        This function has the goal of measuring how for the subject image is from the target image. The subject will have some threshold range that it will
        vary across and the deviation between the subject image (at the current threshold) and the fixed target image (at some fixed thresholds) will be
        measured.
        :return:
        '''
        image_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\MAX_N2Con_3C=1T=0.png"
        three_d_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\N2Con_3C=1T=0.tif"
        vol_test_im_puncta = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\MIP\\CCCP_1C=0T=0.tif", "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MIP\\CCCP_1C=0T=0Option0.tif"]
        cleaned_image = []

        def saveFigure(name):
            plt.savefig(fname="C:\\RESEARCH\\Mitophagy_data\\Time_split\\Presentation Images\\" + name, bbox_inches="tight")
        def saveIm(name, image):
            plt.imsave(fname="C:\\RESEARCH\\Mitophagy_data\\Time_split\\Presentation Images\\" + name, arr=image)

        '''for vtip in range(len(vol_test_im_puncta)):
            vol_puncta_test0 = io.imread(vol_test_im_puncta[vtip])
            green_region = vol_puncta_test0[..., 1]
            max_image = np.max(vol_puncta_test0, axis=-1)
            threshold_regions = np.greater(max_image, green_region).astype(int)
            cleaned_image.append(threshold_regions)
        print(cleaned_image[0].sum()/cleaned_image[1].sum())'''
        #saveIm("overwhelming_puncta.png", np.stack([cleaned_image[0].astype('uint8') * 255, cleaned_image[1].astype('uint8') * 255, np.zeros_like(cleaned_image[0]).astype('uint8')], axis=-1))
        '''io.imshow(np.stack([cleaned_image[0]*255, cleaned_image[1]*255, np.zeros_like(cleaned_image[0])], axis=-1))
        plt.show()'''
        # this is to select the structure for comparison
        mip_image = io.imread(image_path)
        test_im = self._threshold_image(mip_image, 17, 60)
        label_array, number_of_labels = ndi.label(test_im)
        label_list, label_sizes = np.unique(label_array, return_counts=True)
        label_list, label_sizes = label_list[1:], label_sizes[1:]
        above_size = np.nonzero(label_sizes > 180)
        label_list = label_list[above_size]
        filtered_range = np.zeros_like(label_array)
        for n in label_list:
            filtered_range += np.equal(label_array, n).astype(int)
        refined_labels, num_labels = ndi.label(filtered_range)
        structure_selection = (np.nonzero(refined_labels == 42), np.nonzero(refined_labels == 46))
        isolated_structure = (np.equal(refined_labels, 42).astype(int), np.equal(refined_labels, 46).astype(int))
        reduced_max = 160
        secondary_struct_rescale = reduced_max/(isolated_structure[1] * mip_image).max()
        xmin = min(min(structure_selection[0][0]), min(structure_selection[1][0]))
        xmax = max(max(structure_selection[0][0]), max(structure_selection[1][0]))
        ymin = min(min(structure_selection[0][1]), min(structure_selection[1][1]))
        ymax = max(max(structure_selection[0][1]), max(structure_selection[1][1]))
        xrange, yrange = [xmin, xmax], [ymin, ymax]
        '''array_ranges = [xrange[1] - xrange[0], yrange[1] - yrange[0]]
        reduced_canvas = np.zeros(tuple(array_ranges))
        reduced_canvas += (mip_image * isolated_structure[0])[xrange[0]:xrange[1], yrange[0]:yrange[1]]'''
        full_image = io.imread(three_d_path)
        full_ranges = [full_image.shape[0], xrange[1] - xrange[0], yrange[1] - yrange[0]]
        reduced_canvas = np.zeros(tuple(full_ranges))
        reduced_canvas = full_image[:, xrange[0]:xrange[1], yrange[0]:yrange[1]]
        print("3D shape", reduced_canvas.shape)
        # reduced_canvas += (mip_image * isolated_structure[1] * secondary_struct_rescale)[xrange[0]:xrange[1], yrange[0]:yrange[1]].astype(int)
        def test_structure_thresh(low_thresh, high_thresh):
            return self._threshold_image(reduced_canvas, low_thresh, high_thresh)
        subject_thresholds = (33, 48)
        target_thresholds = (22, 62)
        subject_im, target_im = test_structure_thresh(subject_thresholds[0], subject_thresholds[1]), test_structure_thresh(target_thresholds[0], target_thresholds[1])
        '''io.imshow(np.max(target_im, axis=0))
        plt.show()'''
        overlayed_bases = np.stack([subject_im.astype('uint8') * 255, target_im.astype('uint8') * 255, np.zeros_like(subject_im).astype('uint8')], axis=-1)
        '''saveIm("startOverlay_" + "S" + str(subject_thresholds[0]) + "I" + str(subject_thresholds[1]) + "_T" + str(target_thresholds[0]) + "I" +
               str(target_thresholds[1]) + ".png", overlayed_bases)'''
        resolution_minimum = 20
        low_res_steps = math.ceil(abs(target_thresholds[0] - subject_thresholds[0]) / resolution_minimum)
        high_res_steps = math.ceil(abs(target_thresholds[1] - subject_thresholds[1]) / resolution_minimum)
        step_signs = ((target_thresholds[0]-subject_thresholds[0])/abs(target_thresholds[0]-subject_thresholds[0]),
                      (target_thresholds[1]-subject_thresholds[1])/abs(target_thresholds[1]-subject_thresholds[1]))
        low_min = min(subject_thresholds[0], target_thresholds[0])
        low_max = max(subject_thresholds[0], target_thresholds[0])
        high_min = min(subject_thresholds[1], target_thresholds[1])
        high_max = max(subject_thresholds[1], target_thresholds[1])
        low_stop = low_min if step_signs[0] > 0 else low_max
        high_stop = high_min if step_signs[1] > 0 else high_max

        low_res = np.arange(subject_thresholds[0], low_stop + low_res_steps * resolution_minimum * step_signs[0] + step_signs[0], low_res_steps * step_signs[0])
        low_res = np.clip(low_res, low_min, low_max)
        high_res = np.arange(subject_thresholds[1], high_stop + high_res_steps * resolution_minimum * step_signs[1] + step_signs[1], high_res_steps * step_signs[1])
        high_res = np.clip(high_res, high_min, high_max)
        print("Resolutions", low_res, high_res)
        ''' The method above will build the threshold values within the selected resolution granularity and compensates for whether subject thresh is greater
        than target thresh'''
        result_array = np.zeros(tuple([len(low_res), len(high_res)]))
        iteration_order = np.zeros(tuple([len(low_res), len(high_res)]))
        indice_array = np.argwhere(iteration_order == 0).sum(axis=1)/2
        iteration_order[np.nonzero(iteration_order + 1)] = indice_array
        iter_step_size = 0.5
        iteration_ranges = np.arange(0, np.max(iteration_order) + iter_step_size, iter_step_size)
        change_in_similarity = np.zeros(tuple([len(low_res), len(high_res)]))
        subject_exclusions = np.zeros(tuple([len(low_res), len(high_res)]))
        thresholding_combos = np.stack([(np.ones_like(result_array).T * low_res).T, np.ones_like(result_array) * high_res], axis=-1)
        thresholding_combos[..., 0] = np.clip(thresholding_combos[..., 0], 0, thresholding_combos[..., 1])
        target_image = test_structure_thresh(thresholding_combos[-1, -1, 0], thresholding_combos[-1, -1, 1])
        subject_image = test_structure_thresh(thresholding_combos[0, 0, 0], thresholding_combos[0, 0, 1])
        target_labels, target_label_count = ndi.label(target_image)
        total_permutations = iteration_order.shape[0] * iteration_order.shape[1]
        print("Number of iterations:", total_permutations)
        iter_count = 0
        for i in iteration_ranges:
            if iter_step_size == 1:
                viable_elements = np.logical_and(np.less_equal(iteration_order, i), np.greater(iteration_order, i - 1))  # for when the iteration step size is 1
            else:
                viable_elements = np.equal(iteration_order, i)
            current_elements = np.argwhere(viable_elements)
            for ce in current_elements:
                thresh_params = thresholding_combos[ce[0], ce[1]]
                changed_subject = test_structure_thresh(thresh_params[0], thresh_params[1])
                changed_labels, changed_label_count = ndi.label(changed_subject)
                overlap_ratio, exclusion_penalty = self._distance_from_target(changed_labels, target_labels, thresh_params)
                change_in_similarity[ce[0], ce[1]] = overlap_ratio
                subject_exclusions[ce[0], ce[1]] = exclusion_penalty
                iter_count += 1
                print("Iterations remaining:", total_permutations - iter_count)

        change_in_similarity2 = change_in_similarity - subject_exclusions
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
        sns.heatmap(change_in_similarity, xticklabels=high_res.astype(str).tolist(), yticklabels=low_res.astype(str).tolist(), ax=ax1)
        ax1.set_title("Change relative to Target")
        ax1.set_xlabel("High Thresh")
        ax1.set_ylabel("Low Thresh")
        sns.heatmap(subject_exclusions, xticklabels=high_res.astype(str).tolist(), yticklabels=low_res.astype(str).tolist(), ax=ax2)
        ax2.set_title("Structure not in Target")
        ax2.set_xlabel("High Thresh")
        ax2.set_ylabel("Low Thresh")
        sns.heatmap(change_in_similarity2, xticklabels=high_res.astype(str).tolist(), yticklabels=low_res.astype(str).tolist(), ax=ax3)
        ax3.set_title("Change relative to Target after exclusions")
        ax3.set_xlabel("High Thresh")
        ax3.set_ylabel("Low Thresh")
        plt.show()


    def high_and_low_testing(self):
        '''
        This method exists for testing the inheritance of information from some starting thresholding parameters to a set of some final thresholding parameters
        for an image to measure sensitivity.
        :return:
        '''
        image_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MAX_N2Con_3C=1T=0.png"
        mip_image = io.imread(image_path)
        test_im = self._threshold_image(mip_image, 17, 60)
        label_array, number_of_labels = ndi.label(test_im)
        label_list, label_sizes = np.unique(label_array, return_counts=True)
        label_list, label_sizes = label_list[1:], label_sizes[1:]
        above_size = np.nonzero(label_sizes > 180)
        label_list = label_list[above_size]
        filtered_range = np.zeros_like(label_array)
        for n in label_list:
            filtered_range += np.equal(label_array, n).astype(int)
        refined_labels, num_labels = ndi.label(filtered_range)
        structure_selection = np.nonzero(refined_labels == 42)
        isolated_structure = np.equal(refined_labels, 42).astype(int)
        xrange, yrange = [min(structure_selection[0]), max(structure_selection[0])], [min(structure_selection[1]), max(structure_selection[1])]
        array_ranges = [xrange[1] - xrange[0], yrange[1] - yrange[0]]
        reduced_canvas = np.zeros(tuple(array_ranges))
        reduced_canvas = (mip_image * isolated_structure)[xrange[0]:xrange[1], yrange[0]:yrange[1]]
        '''io.imshow(reduced_canvas)
        plt.show()'''
        ending_threshes = (20, 140)
        starting_threshes = (100, 200)

        def test_structure_thresh(low_thresh, high_thresh):
            return self._threshold_image(reduced_canvas, low_thresh, high_thresh)

        steps = (-40, -30)

        starting_image = test_structure_thresh(starting_threshes[0], starting_threshes[1])
        starting_labels, _t = ndi.label(starting_image)
        print("Starting structures", _t)
        low_res = np.arange(starting_threshes[0], ending_threshes[0]+(steps[0]/abs(steps[0])), steps[0])
        high_res = np.arange(starting_threshes[1], ending_threshes[1]+(steps[1]/abs(steps[1])), steps[1])
        print(low_res, high_res)
        result_array = np.zeros(tuple([len(low_res), len(high_res)]))
        thresholding_combos = np.stack([(np.ones_like(result_array).T * low_res).T, np.ones_like(result_array) * high_res], axis=-1)
        iteration_order = np.zeros(tuple([len(low_res), len(high_res)]))
        indice_array = np.argwhere(iteration_order == 0).sum(axis=1)/2
        iteration_order[np.nonzero(iteration_order + 1)] = indice_array
        iter_step_size = 0.5
        iteration_ranges = np.arange(iter_step_size, np.max(iteration_order) + iter_step_size, iter_step_size)
        # result_shape_array = np.zeros(tuple([len(low_res), len(high_res), 2]))  # this is just for testing. Will store the shape of the overlap ratios
        store_neighbours = []  # purely for testing as this is less efficient than arrays and in practice neighbourhood inheritance will happen in loop
        im_shape = list(starting_image.shape)
        parent_images = np.zeros(tuple([1] + im_shape))
        parent_images[0] = starting_image
        parent_key = {"00": 0}
        parent_ratio = {}
        struct_counter = {}
        structure_vol_average = np.zeros_like(iteration_order)
        sorted_start_labels, original_volumes = np.unique(starting_labels, return_counts=True)
        for i in iteration_ranges:
            print(i)
            if iter_step_size == 1:
                viable_elements = np.logical_and(np.less_equal(iteration_order, i), np.greater(iteration_order, i - 1))  # for when the iteration step size is 1
            else:
                viable_elements = np.equal(iteration_order, i)
            previous_iter_elements = np.equal(iteration_order, i - iter_step_size)  # this will be a boolean array for the prior threshold values
            ''' The code below is used to map which preceding threshold combinations are parents of the current combinations. If any element is -1 then
            that is not viable. e.g. for the iterations at 0.5 the only option is [0, 0] but each will also have [-1, 1] or [1, -1] which are not viable'''
            # ***********************
            current_elements = np.argwhere(viable_elements)
            inheritance_array = np.stack([current_elements, current_elements, current_elements], axis=0)
            inheritance_array[1, :, 0] -= 1
            inheritance_array[2, :, 1] -= 1
            store_neighbours.append(inheritance_array)
            def determine_inheritance(iter_of_interest):
                '''
                This inner function is used to determine which prior images (and overlap ratio's) are relevant to the currently thresholded structure while
                minimizing memory consumption (don't store all past threshold images) or computation (if it is thresholded once don't threshold it again)
                :param iter_of_interest: The current thresh iteration to inherit from past
                :return:
                '''
                nonneg = np.all(np.greater_equal(iter_of_interest[1:], 0), axis=0)
                past_structs = iter_of_interest[1:][nonneg]
                return past_structs

            # ***********************
            child_images = np.zeros([inheritance_array.shape[1]] + im_shape)
            child_key = []
            child_ratios = {}
            for tp in range(inheritance_array.shape[1]):
                threshold_params = thresholding_combos[inheritance_array[0, tp, 0], inheritance_array[0, tp, 1]]
                parent_iters = determine_inheritance(inheritance_array[:, tp, :])
                thresholded_image = test_structure_thresh(threshold_params[0], threshold_params[1])
                __unused_labels, structure_count = ndi.label(thresholded_image)
                unique_labels = np.unique(__unused_labels)
                current_iter_key = str(inheritance_array[0, tp, 0]) + str(inheritance_array[0, tp, 1])
                child_key.append(current_iter_key)
                child_images[tp] = thresholded_image
                struct_counter[current_iter_key] = (structure_count, unique_labels)
                for par_it in parent_iters:
                    print("!!", current_iter_key, par_it)
                    print("##", str(par_it[0]) + str(par_it[1]), parent_key, parent_images.shape)
                    parent_im = parent_images[parent_key[str(par_it[0]) + str(par_it[1])]]
                    over_ratio, vol_ratio, structure_pairs, excluded = self._structure_overlap(parent_im, thresholded_image)
                    print("Excluded structures", excluded, over_ratio.shape)
                    print(len(parent_ratio.keys()), parent_ratio.keys(), str(par_it[0]) + str(par_it[1]))
                    if len(parent_ratio.keys()) > 0 and str(par_it[0]) + str(par_it[1]) in parent_ratio:
                        if current_iter_key not in child_ratios:
                            child_ratios[current_iter_key] = []
                        print("Ratio shapes", parent_ratio[str(par_it[0]) + str(par_it[1])].shape, over_ratio.shape)
                        ratio_transfer = np.matmul(parent_ratio[str(par_it[0]) + str(par_it[1])], over_ratio)
                        child_ratios[current_iter_key].append(ratio_transfer)
                        non_zero_structs = np.nonzero(over_ratio)
                        print("1111111111111111111111111111111111111111112")
                        print(np.unique(non_zero_structs[0]).shape, np.unique(non_zero_structs[1]).shape)
                        print(over_ratio)
                        if len(excluded[2]) > 1:
                            print("Structures in new", structure_count)
                            parent_volumes = (parent_ratio[str(par_it[0]) + str(par_it[1])].T * original_volumes)
                            overlap_struct_volumes = np.matmul(parent_volumes, ratio_transfer)
                            par_labelled, _p = ndi.label(parent_im)
                            child_labelled, _c = ndi.label(thresholded_image)
                            par_labels, parent_vols = np.unique(par_labelled, return_counts=True)
                            chi_labels, chi_vols = np.unique(child_labelled, return_counts=True)
                            print("Apparent overlap volumes", overlap_struct_volumes)
                            print('Parent & Child volumes', parent_vols[1:], chi_vols[1:])
                            print("Original volumes", original_volumes[1:])
                            print("Parent volume consolidated", (parent_ratio[str(par_it[0]) + str(par_it[1])].T * original_volumes).T.sum(axis=1))
                            print("Child volume consolidated", (ratio_transfer.T * original_volumes).T.sum(axis=1))
                            io.imshow(np.stack([par_labelled * 10, child_labelled * 10, np.zeros_like(thresholded_image)], axis=-1))
                            plt.show()
                        print("2111111111111111111111111111111111111111111")
                    else:
                        child_ratios[current_iter_key] = [over_ratio]
                if len(child_ratios[current_iter_key]) > 1:
                    child_ratios[current_iter_key] = (child_ratios[current_iter_key][0] + child_ratios[current_iter_key][1])/2
                else:
                    child_ratios[current_iter_key] = child_ratios[current_iter_key][0]
                print("Child shape chase:", child_ratios[current_iter_key].shape)
                child_volumes = (child_ratios[current_iter_key].T * original_volumes).T
                consolidated_overlaps = (child_ratios[current_iter_key] * child_volumes).sum(axis=1)
                print("~~~~", current_iter_key, consolidated_overlaps)
                structure_vol_average[inheritance_array[0, tp, 0], inheritance_array[0, tp, 1]] = np.mean(consolidated_overlaps[1:])
            parent_key = {}
            parent_images = np.zeros_like(child_images)
            parent_ratio = {}
            for k, v in child_ratios.items():
                print("Child", k)
                print(v.shape)

            for ck in range(len(child_key)):
                parent_key[child_key[ck]] = ck
                parent_images[ck] = child_images[ck]
                parent_ratio[child_key[ck]] = child_ratios[child_key[ck]]


        print("Structures at each iteration")
        print(struct_counter)
        print(structure_vol_average)
        '''print("Going through neighbours")

        def result_shape_testing(neighbourhoods):
            structure_current = []
            for y in range(0, neighbourhoods.shape[1]):
                rows, columns = [], []
                has_neighbour = False
                for t in [1, 2]:
                    non_negative = np.all(np.greater_equal(neighbourhoods[t, y, :], 0))
                    if non_negative:
                        has_neighbour = True
                        rows.append(neighbourhoods[t, y, 0])
                        columns.append(neighbourhoods[t, y, 1])
                if has_neighbour:
                    rows.append(neighbourhoods[0, y, 0])
                    columns.append(neighbourhoods[0, y, 1])
                structure_current.append([np.array(rows), np.array(columns)])
            return structure_current

        for sn in store_neighbours:
            neighbouring_structs = result_shape_testing(sn)
            for sh in neighbouring_structs:
                print("**************************")
                print(result_shape_array[sh[0], sh[1]])
                print("**************************")
            print("#################################")

        print("Manual check")

        starting_image = test_structure_thresh(starting_threshes[0], starting_threshes[1])  # resetting for a manual check
        image_01 = test_structure_thresh(thresholding_combos[0, 1][0], thresholding_combos[0, 1][1])
        over_0001, __a, __b, __c = self._structure_overlap(starting_image, image_01)
        image_10 = test_structure_thresh(thresholding_combos[0, 1][0], thresholding_combos[0, 1][1])
        over_0010, __a, __b, __c = self._structure_overlap(starting_image, image_10)
        image_11 = test_structure_thresh(thresholding_combos[1, 1][0], thresholding_combos[1, 1][1])

        over_0111, __a, __b, __c = self._structure_overlap(image_01, image_11)
        over_1011, __a, __b, __c = self._structure_overlap(image_10, image_11)

        image_02 = test_structure_thresh(thresholding_combos[0, 2][0], thresholding_combos[0, 2][1])
        image_20 = test_structure_thresh(thresholding_combos[2, 0][0], thresholding_combos[2, 0][1])
        over_0102, __a, __b, __c = self._structure_overlap(image_01, image_02)
        over_1020, __a, __b, __c = self._structure_overlap(image_10, image_20)

        image_12 = test_structure_thresh(thresholding_combos[1, 2][0], thresholding_combos[1, 2][1])
        image_21 = test_structure_thresh(thresholding_combos[2, 1][0], thresholding_combos[2, 1][1])

        over_0212, __a, __b, __c = self._structure_overlap(image_02, image_12)
        over_1112, __a, __b, __c = self._structure_overlap(image_11, image_12)

        over_1121, __a, __b, __c = self._structure_overlap(image_11, image_21)
        over_2021, __a, __b, __c = self._structure_overlap(image_20, image_21)

        print(over_0001.shape, over_0010.shape)
        print(over_0111.shape, over_1011.shape)

        print("*******************")
        print(over_0102.shape, over_1020.shape)
        print(over_0212.shape, over_1112.shape, "#", over_1121.shape, over_2021.shape)
        image_22 = test_structure_thresh(thresholding_combos[2, 2][0], thresholding_combos[2, 2][1])
        over_1222, __a, __b, __c = self._structure_overlap(image_12, image_22)
        over_2122, __a, __b, __c = self._structure_overlap(image_21, image_22)
        print(over_1222.shape, over_2122.shape)

        def adjust_values_for_pairs(arr1, arr2):
            bool1 = np.greater(arr1, 0).astype(int)
            bool2 = np.greater(arr2, 0).astype(int)
            col_range = bool2.shape[1]
            col_numbers = np.arange(0, col_range, 1)
            bool2 = bool2 * col_numbers
            result = np.matmul(bool1, bool2)
            old_structs = result.shape[0]
            stored = {}
            for os in range(1, old_structs):
                stored[os] = np.where(result[os] > 0)[0].tolist()
            return result, stored

        def relative_traceback(prior, current):
            # print("###**", prior.shape, current.shape, "**###")
            return np.matmul(prior, current)

        over_02, store02 = adjust_values_for_pairs(over_0001, over_0102)
        over_20, store20 = adjust_values_for_pairs(over_0010, over_1020)
        over_11a, store11a = adjust_values_for_pairs(over_0001, over_0111)
        over_11b, store11b = adjust_values_for_pairs(over_0010, over_1011)
        print("###########################")
        print(over_11a.sum(axis=1))
        print(over_11b.sum(axis=1))
        print("**************************")
        over_21a, store21a = adjust_values_for_pairs(over_20, over_2021)
        over_21b, store21b = adjust_values_for_pairs(over_11a, over_1121)

        over_12a, store12a = adjust_values_for_pairs(over_02, over_0212)
        over_12b, store12b = adjust_values_for_pairs(over_11a, over_1112)

        print(over_21a.sum(axis=1), over_21b.sum(axis=1))
        print("**************************")
        print(over_12a.sum(axis=1), over_12b.sum(axis=1))
        over_02 = relative_traceback(over_0001, over_0102)
        over_20 = relative_traceback(over_0010, over_1020)
        over_11a = relative_traceback(over_0001, over_0111)
        over_11b = relative_traceback(over_0010, over_1011)
        print("###########################")
        print(over_11a)
        print(over_11b)
        print("**************************")
        over_21a = relative_traceback(over_20, over_2021)
        over_21b = relative_traceback(over_11a, over_1121)

        over_12a = relative_traceback(over_02, over_0212)
        over_12b = relative_traceback(over_11a, over_1112)
        print(over_21a)
        print(over_21b)
        print("**************************")
        print(over_12a)
        print(over_12b)
        print(over_0001.shape, over_0111.shape, over_0102.shape, over_0212.shape, over_1112.shape, over_1222.shape, over_0010.shape, over_1011.shape,
              over_1020.shape, over_1121.shape, over_2021.shape, over_2122.shape)'''




    def structure_hunting(self):
        image_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\MAX_N2Con_3C=1T=0.png"
        mip_image = io.imread(image_path)
        test_im = self._threshold_image(mip_image, 17, 60)
        label_array, number_of_labels = ndi.label(test_im)
        label_list, label_sizes = np.unique(label_array, return_counts=True)
        label_list, label_sizes = label_list[1:], label_sizes[1:]
        above_size = np.nonzero(label_sizes > 180)
        label_list = label_list[above_size]
        filtered_range = np.zeros_like(label_array)
        for n in label_list:
            filtered_range += np.equal(label_array, n).astype(int)
        refined_labels, num_labels = ndi.label(filtered_range)
        structure_selection = np.nonzero(refined_labels == 42)
        isolated_structure = np.equal(refined_labels, 42).astype(int)
        padding = 10
        xrange, yrange = [min(structure_selection[0]), max(structure_selection[0])], [min(structure_selection[1]), max(structure_selection[1])]
        array_ranges = [xrange[1]-xrange[0], yrange[1]-yrange[0]]
        reduced_canvas = np.zeros(tuple(array_ranges))
        reduced_canvas = (mip_image*isolated_structure)[xrange[0]:xrange[1], yrange[0]:yrange[1]]
        '''io.imshow(reduced_canvas)
        plt.show()'''
        starting_low = 20
        thresholded_canvas = np.zeros_like(reduced_canvas)
        '''for t in range(starting_low, starting_low*5+1, 20):
            thresholded_canvas += self._threshold_image(reduced_canvas, t, 140)'''

        low_start = 60
        def test_structure_thresh(low_thresh):
            return self._threshold_image(reduced_canvas, low_thresh, 140)



        print("##################################\n############ 40 to 60 ############\n##################################")
        volume40to60, pairs40to60 = self._structure_overlap(test_structure_thresh(40), test_structure_thresh(60))
        print("##################################\n############ 60 to 80 ############\n##################################")
        volume60to80, pairs60to80 = self._structure_overlap(test_structure_thresh(60), test_structure_thresh(80))
        print("##################################\n############ 80 to 100 ###########\n##################################")
        volume80to100, pairs80to100 = self._structure_overlap(test_structure_thresh(80), test_structure_thresh(100))
        print("##################################\n############ 100 to 120 ###########\n##################################")
        volume100to120, pairs100to120 = self._structure_overlap(test_structure_thresh(100), test_structure_thresh(120))

        print(volume40to60)

        def adjust_values_for_pairs(arr1, arr2):
            bool1 = np.greater(arr1, 0).astype(int)
            bool2 = np.greater(arr2, 0).astype(int)
            col_range = bool2.shape[1]
            #print(col_range)
            col_numbers = np.arange(0, col_range, 1)
            bool2 = bool2 * col_numbers
            result = np.matmul(bool1, bool2)
            old_structs = result.shape[0]
            stored = {}
            for os in range(1, old_structs):
                stored[os] = np.where(result[os] > 0)[0].tolist()
            return result, stored

        def convert_vol_ratios(arr1, arr2):
            bool1 = arr1
            bool2 = arr2
            result = np.matmul(bool1, bool2)
            return result

        print("##################################\n############ 60 to 40 ############\n##################################")
        volume60to40, pairs60to40 = self._structure_overlap(test_structure_thresh(60), test_structure_thresh(40))
        print("##################################\n############ 80 to 60 ############\n##################################")
        volume80to60, pairs80to60 = self._structure_overlap(test_structure_thresh(80), test_structure_thresh(60))
        print("##################################\n############ 100 to 80 ###########\n##################################")
        volume100to80, pairs100to80 = self._structure_overlap(test_structure_thresh(100), test_structure_thresh(80))
        print("Tracking changes forwards")
        change1, store1 = adjust_values_for_pairs(volume40to60, volume60to80)
        print(pairs40to60, pairs60to80, "!", store1)
        change2, store2 = adjust_values_for_pairs(volume60to80, volume80to100)
        print(pairs60to80, pairs80to100, "!", store2)
        change3, store3 = adjust_values_for_pairs(volume80to100, volume100to120)
        print(pairs80to100, pairs100to120, "!", store3)
        '''print("In reverse")
        r_change1, r_store1 = adjust_values_for_pairs(volume100to80, volume80to60)
        print(pairs100to80, pairs80to60)
        r_change2, r_store2 = adjust_values_for_pairs(volume80to60, volume60to40)
        print(pairs80to60, pairs60to40)'''

        print("##########################")
        '''print(volume40to60, volume60to80, volume80to100)
        print(change1.shape, change2.shape, change3.shape)
        print(change1)
        print(change2)
        AtoC, new_store1 = adjust_values_for_pairs(change1, change3)
        print(AtoC, new_store1)
        old = volume40to60
        print("Initial volume:\n", old)
        image_set = [volume60to80, volume80to100, volume100to120]
        final_change, final_pairings = None, None
        for timage in range(len(image_set)):
            changed_ver = convert_vol_ratios(old, image_set[timage])
            print(old.shape, image_set[timage].shape, changed_ver.shape)
            print("Step:", timage, "Ratios:\n", changed_ver.sum(axis=1))
            old = changed_ver
            final_change = changed_ver
        print("************************")
        print(final_change)
        print("########################")
        print(volume40to60)
        print(volume60to80)
        print(volume80to100)'''

        def step_to_step_ratio_relative(distant_past, prior, current, volumes):
            past_volumes = (distant_past.T * volumes).T
            prior_volumes = np.matmul(past_volumes, prior)
            current_volumes = np.matmul(prior_volumes, current)
            binary_past = np.greater(distant_past, 0).astype(int)
            binary_prior = np.greater(prior, 0).astype(int)
            print(distant_past.shape, prior.shape, current.shape)
            print(current)
            print("****************************")
            print(prior)
            print("############ Reshaped")
            reshaped_to_prior = np.matmul(binary_prior, current)
            print(reshaped_to_prior.sum(axis=1))
            reshaped_to_past = np.matmul(binary_past, reshaped_to_prior)
            prior_reshape = np.matmul(binary_past, prior)
            print(reshaped_to_past[1:])
            print("************************************")
            print(prior_reshape[1:])
            print(reshaped_to_past[1:].mean(axis=1), prior_reshape[1:].mean(axis=1))
            print(reshaped_to_past[1:].mean(axis=1, where=reshaped_to_past[1:] > 0), prior_reshape[1:].mean(axis=1, where=prior_reshape[1:] > 0))
            print("#*#*#*#*")
            print(reshaped_to_past.sum(axis=1), prior_reshape.sum(axis=1))
            reshaped_truth = np.matmul(prior, current)
            reshaped_to_past = np.matmul(binary_past, reshaped_truth)
            print(reshaped_to_past.sum(axis=1))
            print("Volume Comparison")
            print(past_volumes.sum(axis=1))
            print(prior_volumes.sum(axis=1))
            print(current_volumes.sum(axis=1))


        sub_label_array, _label_count = ndi.label(test_structure_thresh(100) > 0)
        subject_labels, subject_volumes = np.unique(sub_label_array, return_counts=True)
        step_to_step_ratio_relative(volume100to80, volume80to60, volume60to40, subject_volumes)


    def _structure_overlap_test(self):
        im1 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
               [1, 1, 1, 1, 0, 0, 1, 1],
               [1, 1, 1, 1, 0, 0, 1, 1],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               ])
        im2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [0, 1, 1, 1, 1, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               ])
        self._structure_overlap(im1, im2)
        im3 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        ])
        im4 = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        ])
        im5 = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        ])
        im6 = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        ])
        self._structure_overlap(im4, im3)
        print("Swapped")
        self._structure_overlap(im3, im4)
        print("Bridged")
        self._structure_overlap(im4, im5)
        print("More structures")
        self._structure_overlap(im5, im6)

    def _image_change_calculation(self, orig_image, threshes1, threshes2, step=5):
        '''
        This method will track the structure based changes across an image and the overlap between said image and a reference image as the thresholds are
        tended towards the reference. The comparison of an image with it's prior self can provide reference for the spatial change relative to the original
        thresholds specifically on the impact of both the low and high thresholds respectively. The reference image will be fixed and will be used to map
        overlap changes from the adjusted image.
        :param orig_image:
        :param threshes1:
        :param threshes2:
        :return:
        '''
        im1_threshed = self._threshold_image(orig_image, threshes1[0], threshes1[1])
        im2_threshed = self._threshold_image(orig_image, threshes2[0], threshes2[1])
        low_resolution = math.ceil((abs(threshes1[0] - threshes2[0]) / step))
        high_resolution = math.ceil((abs(threshes1[1] - threshes2[1]) / step))
        # the resolutions will be used to build the storage mapping. For now this will store the change in an array of low_res x high_res
        distance_log = np.zeros(tuple([low_resolution, high_resolution]))
        # will be a single loop where high_counter=0 and distance_log[n, high_counter*step*n - n
        offset = 0
        print("Thresholds:", threshes1, threshes2)
        print("Resolutions:", low_resolution, high_resolution)

        def threshold_adjustment(thr1, thr2, low_adjust, high_adjust):
            low_res_sh = low_adjust + thr1[0]
            high_res_sh = high_adjust + thr1[1]
            if (thr1[0] > thr2[0] and low_res_sh >= thr2[0]) or (thr1[0] < thr2[0] and low_res_sh <= thr2[0]):
                low_r = low_res_sh
            else:
                low_r = thr2[0]

            if (thr1[1] > threshes2[1] and high_res_sh >= threshes2[1]) or (thr1[1] < threshes2[1] and high_res_sh <= threshes2[1]):
                high_r = high_res_sh
            else:
                high_r = thr2[1]
            return low_r, high_r

        for resol in range(low_resolution + high_resolution):
            if resol - offset*high_resolution > high_resolution:
                offset += 1  # this will be used to constrain the value in the distance_log indices to remain in range
            print("Current Resolution", resol)
            low_res_adjust = step * (resol - offset * high_resolution)  # this should return the correct low resolution
            high_res_adjust = step * offset  # the high resolution will be the second loop typically
            if threshes1[0] > threshes2[0]:
                low_res_adjust *= -1
            if threshes1[1] > threshes2[1]:
                high_res_adjust *= -1
            #  if the target (expert) has thresholds then the change must tend in that direction
            low_res, high_res = threshold_adjustment(threshes1, threshes2, low_res_adjust, high_res_adjust)
            print(low_res, high_res)

    def _image_difference_calculation(self, im1, im2, overlapping_structures):
        '''
        This method will calculate the relative volume of the overlapping regions to the root structure, compare this percentage overlap between them and
        their volumes relative to each other. im1 and im2 must only contain the structures that are partially overlapping. Overlapping structures should
        be a list of tuples with the first tuple element being the im1 struct and the second element being the im2 struct
        :param im1:
        :param im2:
        :param overlapping_structures:
        :return:
        '''
        pairing_results = {}
        for os in overlapping_structures:
            im1_struct = (im1 == os[0]).astype('uint8')
            im2_struct = (im2 == os[1]).astype('uint8')
            overlap_seg = np.logical_and(im1_struct, im2_struct)
            im1_rel_overlap = (im1_struct * overlap_seg).sum() / im1_struct.sum()  # im1 overlap ratio
            im2_rel_overlap = (im2_struct * overlap_seg).sum() / im2_struct.sum()  # im2 overlap ratio
            average_struct_size = ((im1_struct.sum() + im2_struct.sum())/2)  # needs to take all of the currently overlapping structs since all in overlap zone


    def _image_analysis(self, image1, image2):
        '''
        This method will compare the visual differences between the images provided. The images will be the intensity images.
        :param image1:
        :param image2:
        :return:
        '''
        voxel_count_difference = np.count_nonzero(image1) - np.count_nonzero(image2)
        self._structure_overlap(image1, image2)
        if voxel_count_difference > 0:
            larger_image = 1
        elif voxel_count_difference < 0:
            larger_image = 2
        else:
            larger_image = 0
        voxel_intensity_difference = abs(image1 - image2)

    def _ordered_mapping_overlaps(self, structure_labels):
        '''
        This method will return an array that maps the overlap region structure numbers to the reduced list of actual overlapping structures
        :param structure_labels: unique list of integers which represent the present structures
        :return: A dictionary with the keys being the structure labels and the value being their order in the structure_labels list
        '''
        label_maps = {}
        for i in range(len(structure_labels)):
            label_maps[structure_labels[i]] = i
        return label_maps

    def _preview_overlap_isolated(self, overlap1, overlap2):
        zero_temp = np.zeros_like(overlap1)
        template = np.stack([overlap1, overlap2, zero_temp], axis=-1)
        projected_2d = np.amax(template, axis=0)
        io.imshow(projected_2d)
        plt.show()

    def _structure_overlap_effi(self, image1, image2, labels_provided=False):
        '''
        This method is designed to determine what labelled structures from each of the images overlap. From here the percentage overlap relative to the
        overlapping structures complete volumes could be calculated as well as a relationship between structure aggregation? (one large overlaps with many
        small). When looking at historical change then image1 must be old and image2 new threshed version
        Have im1 and im2 already be labeled arrays instead. This way it is known that the labels will correspond when passed into this function for future
        iterations.
        :param image1: subject image that calculations are in reference to
        :param image2: reference image that the subject is compared to
        :return: over_ratio, vol_ratio, subject_match_relations, excluded_structures
        '''

        overlap_image = np.logical_and(np.greater(image1, 0), np.greater(image2, 0))
        excluded_structures = np.logical_not(overlap_image, where=np.logical_or(np.greater(image1, 0), np.greater(image2, 0))).astype('uint8')
        # the where argument designates where the non-zero regions (not background) are for either image
        # if the logic below fails then it means that all structures in both images are perfectly aligned and equal in shape, volume and count
        # self._composite_image_preview(image1, image2, overlap_image)
        overlap_regions, overlap_count = ndi.label(overlap_image)
        if not labels_provided:
            binary1 = image1 > 0
            binary2 = image2 > 0
            structure_seg1, structure_count1 = ndi.label(binary1)  # this labeled array should be an argument
            structure_seg2, structure_count2 = ndi.label(binary2)  # same for this labeled array
        else:
            structure_seg1, structure_seg2 = image1, image2
            structure_count1, structure_count2 = len(np.unique(structure_seg1)[1:]), len(np.unique(structure_seg2)[1:])
        valid_structures = np.stack([structure_seg1 * overlap_image, structure_seg2 * overlap_image], axis=-1)
        reordered = np.reshape(valid_structures, (-1, valid_structures.shape[-1]))
        print("Reordeded shape", reordered.shape, valid_structures.shape)
        invalid_pairs = np.logical_not(np.any(reordered == 0, axis=-1))
        pairings, pairing_sizes = np.unique(reordered[invalid_pairs], return_counts=True, axis=0)
        pairings_list = np.split(pairings.T, indices_or_sections=2, axis=0)
        paired_structures = np.zeros(tuple([structure_count1 + 1, structure_count2 + 1]))
        # pairing_array = pairing_sizes[:, np.newaxis]
        paired_structures[pairings_list[0][0], pairings_list[1][0]] = pairing_sizes
        overlapped_structures = np.nonzero(paired_structures)  # this will return the structure pairs that overlap.
        subject_im_labels, subject_im_volumes = np.unique(structure_seg1, return_counts=True)  # will return the struct label list for im1 and the volumes
        ref_im_labels, ref_im_volumes = np.unique(structure_seg2, return_counts=True)
        def _overlap_ratios():
            included_subject = overlapped_structures[0]  # the first dim in paired_structures is for the subject image struct labels
            included_reference = overlapped_structures[1]
            excluded_subj_structs = np.setdiff1d(subject_im_labels, included_subject)  # this should return the labels of the ignored structures
            excluded_ref_structs = np.setdiff1d(ref_im_labels, included_reference)
            excluded_struct_vol = (excluded_subj_structs, subject_im_volumes[excluded_subj_structs], excluded_ref_structs, ref_im_volumes[excluded_ref_structs])
            over_ratio = np.zeros_like(paired_structures)
            over_ratio[overlapped_structures] = 1
            over_ratio[included_subject] *= np.divide(paired_structures[included_subject].T, subject_im_volumes[included_subject]).T  # the transpose is done for broadcasting. 0 < ratio <= 1
            over_ratio[:, included_reference] *= np.divide(paired_structures[:, included_reference], ref_im_volumes[included_reference])
            vol_ratio = (paired_structures > 0).astype(float)
            vol_ratio[included_subject] = (vol_ratio[included_subject].T * subject_im_volumes[included_subject]).T
            vol_ratio[:, included_reference] = (vol_ratio[:, included_reference] / ref_im_volumes[included_reference])
            ''' the np.divide approach could be used but where the ratio is no transposed back to get the overlap percentage relative to the reference.
            The value of this is that the distance of the subject from the reference should also take into account the fully excluded structures on each side.
            '''
            subject_match_relations = {a: [] for a in np.unique(included_subject)}  # this will make the pair storage dictionary
            pair_wise = np.argwhere(paired_structures)
            for pw in pair_wise:
                subject_match_relations[pw[0]].append(pw[1])
            ''' The below will return the ratio of overlapped volume relative to the complete volume of the original/prior structure. The vol_ratio is the 
            ratio between the old image and the new image. The subject_match_relations is a dictionary for the currently overlapping structures.
            excluded_structs is a tuple with the first elem for the im1 structs not in im2 and the second elem is for the im2 structs not in im1'''
            return over_ratio, vol_ratio, subject_match_relations, excluded_struct_vol

        return _overlap_ratios()

    def _structure_overlap(self, image1, image2, labels_provided=False):
        '''
        This method is designed to determine what labelled structures from each of the images overlap. From here the percentage overlap relative to the
        overlapping structures complete volumes could be calculated as well as a relationship between structure aggregation? (one large overlaps with many
        small). When looking at historical change then image1 must be old and image2 new threshed version
        Have im1 and im2 already be labeled arrays instead. This way it is known that the labels will correspond when passed into this function for future
        iterations.
        :param image1: subject image that calculations are in reference to
        :param image2: reference image that the subject is compared to
        :return: over_ratio, vol_ratio, subject_match_relations, excluded_structures
        '''

        overlap_image = np.logical_and(np.greater(image1, 0), np.greater(image2, 0))
        excluded_structures = np.logical_not(overlap_image, where=np.logical_or(np.greater(image1, 0), np.greater(image2, 0))).astype('uint8')
        # the where argument designates where the non-zero regions (not background) are for either image
        # if the logic below fails then it means that all structures in both images are perfectly aligned and equal in shape, volume and count
        # self._composite_image_preview(image1, image2, overlap_image)
        overlap_regions, overlap_count = ndi.label(overlap_image)
        if not labels_provided:
            binary1 = image1 > 0
            binary2 = image2 > 0
            structure_seg1, structure_count1 = ndi.label(binary1)  # this labeled array should be an argument
            structure_seg2, structure_count2 = ndi.label(binary2)  # same for this labeled array
        else:
            structure_seg1, structure_seg2 = image1, image2
            structure_count1, structure_count2 = len(np.unique(structure_seg1)[1:]), len(np.unique(structure_seg2)[1:])
        starting_time1 = time.process_time_ns()
        valid_structures = np.stack([structure_seg1 * overlap_image, structure_seg2 * overlap_image], axis=-1)
        reordered = np.reshape(valid_structures, (-1, valid_structures.shape[2]))
        print("Flattened shape", reordered.shape)
        invalid_pairs = np.logical_not(np.any(reordered == 0, axis=-1))
        pairings, pairing_sizes = np.unique(reordered[invalid_pairs], return_counts=True, axis=0)
        end_time1 = time.process_time_ns()
        print("Time1", end_time1 - starting_time1)
        # pairing_array = pairing_sizes[:, np.newaxis]
        # print("Pairings to volumes \n", np.append(pairings, pairing_array, axis=-1))
        # below will retrieve the labels of the image structures that are within the overlap regions
        im1_overlap_structs = np.unique(structure_seg1 * overlap_image).tolist()
        im1_overlap_structs.remove(0)
        im2_overlap_structs = np.unique(structure_seg2 * overlap_image).tolist()
        im2_overlap_structs.remove(0)
        # below is commented out. Currently all overlapping structures (partial or complete) will be tracked for historical comparison
        im1_excluded_structs = np.unique(structure_seg1 * excluded_structures).tolist()
        im1_excluded_structs.remove(0)
        im2_excluded_structs = np.unique(structure_seg1 * excluded_structures).tolist()
        im2_excluded_structs.remove(0)
        # below will determine the structures that are currently alone with no overlapping structures
        lonely_structs1 = set(im1_excluded_structs).difference(im1_overlap_structs)
        lonely_structs2 = set(im2_excluded_structs).difference(im2_overlap_structs)
        '''
        # This will look to see if there are any matched structures that spill out of the overlap region
        im1_mismatch = set(im1_overlap_structs).isdisjoint(im1_excluded_structs)
        im2_mismatch = set(im2_overlap_structs).isdisjoint(im2_excluded_structs)
        print("Disjoint check:", im1_mismatch, im2_mismatch)
        print("Partially overlapping structures:", set(im1_overlap_structs).intersection(im1_excluded_structs),
              set(im2_overlap_structs).intersection(im2_excluded_structs))'''
        # This will be an array for the structure pairings. Currently the array will contain all structures (not just the overlapping ones)
        paired_structures = np.zeros(tuple([structure_count1+1, structure_count2+1]))
        '''io.imshow(np.amax(np.stack([structure_seg1*overlap_image, structure_seg2*overlap_image, np.zeros_like(overlap_image)], axis=-1), axis=0))
        plt.show()'''
        print("There are", overlap_count, " overlapping regions")
        overlap_pair_volume_shared = np.zeros(tuple([len(im1_overlap_structs), len(im2_overlap_structs)]))
        start_time2 = time.process_time_ns()
        for over_regions in range(1, overlap_count+1):
            isolated_overlap = np.equal(overlap_regions, over_regions).astype('uint8')
            '''io.imshow(isolated_overlap)
            plt.show()'''
            image1_overlap = structure_seg1 * isolated_overlap
            image2_overlap = structure_seg2 * isolated_overlap
            image1_label, im1_volumes = np.unique(image1_overlap, return_counts=True)
            image2_label, im2_volumes = np.unique(image2_overlap, return_counts=True)
            nonzero1 = np.greater(image1_label, 0)
            nonzero2 = np.greater(image2_label, 0)
            if np.any(nonzero1) and np.any(nonzero2):
                image1_label = image1_label[nonzero1]  # removes 0 (background) label from list using boolean indexing
                im1_volumes = im1_volumes[nonzero1]  # using the same method the volume corresponding to the background is removed
                image2_label = image2_label[nonzero2]
                im2_volumes = im2_volumes[nonzero2]
                # print(image1_label, image2_label, im1_volumes, im2_volumes)
                ''' Below used to be used to map the structure labels to the limit range only overlapping labels. This has now been adjusted and the pairing
                array will take all structures for each image and so this mapping is no longer needed.'''
                '''im1_mapped = [im1_mapping[i1] for i1 in image1_label]
                im2_mapped = [im2_mapping[i2] for i2 in image2_label]'''
                paired_structures[tuple([tuple(image1_label.tolist()), tuple(image2_label.tolist())])] += (im1_volumes + im2_volumes)/2
                ''' paired structures is a mapping of the mean volumes shared. It should be that mean_vol == im1_vol == im2_vol but the mean compensates'''
        ''' The change to paired_structures indexing across the full structure list means that when reading these nonzero tuples they directly can
        correspond to the respective image labels as opposed to requiring a mapping function to reverse the _ordered_mapping_overlaps relationship'''
        # print("Structure pairs", paired_structures)
        end_time2 = time.process_time_ns()
        print("Time2", end_time2 - start_time2)
        overlapped_structures = np.nonzero(paired_structures)  # this will return the structure pairs that overlap.
        print("Overlapping Structures")
        print(paired_structures)
        subject_im_labels, subject_im_volumes = np.unique(structure_seg1, return_counts=True)  # will return the struct label list for im1 and the volumes
        ref_im_labels, ref_im_volumes = np.unique(structure_seg2, return_counts=True)
        '''non_zero = np.greater(subject_im_labels, 0)
        subject_im_labels, subject_im_volumes = subject_im_labels[non_zero], subject_im_volumes[non_zero]'''
        ''' With this the structure labels that experience overlap will be stored in overlapping pairs, using overlapping pairs their values can be 
        extracted (volumes) and subject_im_volumes can be used for percentage overlap. Prior image must be an optional argument for the reference image
        for mapping. Must be None (default) or a positional mapping '''
        # print("Volumes:", subject_im_labels, subject_im_volumes, "\n", ref_im_labels, ref_im_volumes)
        # print("Shared volumes:", paired_structures)
        def _overlap_ratios():
            included_subject = overlapped_structures[0]  # the first dim in paired_structures is for the subject image struct labels
            included_reference = overlapped_structures[1]
            excluded_subj_structs = np.setdiff1d(subject_im_labels, included_subject)  # this should return the labels of the ignored structures
            excluded_ref_structs = np.setdiff1d(ref_im_labels, included_reference)
            excluded_struct_vol = (excluded_subj_structs, subject_im_volumes[excluded_subj_structs], excluded_ref_structs, ref_im_volumes[excluded_ref_structs])
            over_ratio = np.zeros_like(paired_structures)
            over_ratio[overlapped_structures] = 1
            over_ratio[included_subject] *= np.divide(paired_structures[included_subject].T, subject_im_volumes[included_subject]).T  # the transpose is done for broadcasting. 0 < ratio <= 1
            over_ratio[:, included_reference] *= np.divide(paired_structures[:, included_reference], ref_im_volumes[included_reference])
            vol_ratio = (paired_structures > 0).astype(float)
            vol_ratio[included_subject] = (vol_ratio[included_subject].T * subject_im_volumes[included_subject]).T
            vol_ratio[:, included_reference] = (vol_ratio[:, included_reference] / ref_im_volumes[included_reference])
            ''' the np.divide approach could be used but where the ratio is no transposed back to get the overlap percentage relative to the reference.
            The value of this is that the distance of the subject from the reference should also take into account the fully excluded structures on each side.
            '''
            subject_match_relations = {a: [] for a in np.unique(included_subject)}  # this will make the pair storage dictionary
            pair_wise = np.argwhere(paired_structures)
            for pw in pair_wise:
                subject_match_relations[pw[0]].append(pw[1])
            ''' The below will return the ratio of overlapped volume relative to the complete volume of the original/prior structure. The vol_ratio is the 
            ratio between the old image and the new image. The subject_match_relations is a dictionary for the currently overlapping structures.
            excluded_structs is a tuple with the first elem for the im1 structs not in im2 and the second elem is for the im2 structs not in im1'''
            return over_ratio, vol_ratio, subject_match_relations, excluded_struct_vol

        '''plt.figure(1)
        io.imshow(structure_seg1)
        plt.figure(2)
        io.imshow(structure_seg2)
        plt.figure(3)
        io.imshow(overlap_image.astype('uint8')*255)
        plt.show()'''
        return _overlap_ratios()

    def _consolidate_prior_structures(self, volumes, pairings):
        '''
        Method to consolidate which structures are newly split/joined, to measure said split or join and to consolidate the volumes accordingly
        :param volumes:
        :param pairings:
        :return:
        '''
        ''' np.greater(volumes, 0).astype(int).sum(axis=n) where for axis=n when n=0 then will measure splits while n=1 measures joins '''
        pairing_count = np.greater(volumes, 0).astype(int)
        split_check = pairing_count.sum(axis=0)
        join_check = pairing_count.sum(axis=1)
        ''' The split_check.sum() == join_check.sum() will hold true since the total number of pairing are the same, just some of the pairings share structs. 
        To deal with this take the np.count_nonzero of each and if mismatched then it is known that a split or join occurred. Check for pairs that are
        greater than one and thus have either split or joined. If np.count_nonzero for axis=1 > axis=0 then a split occurred else a join occurred'''
        print(join_check, split_check)
        split_structures = np.greater(split_check, 1)
        join_structures = np.greater(join_check, 1)
        if np.count_nonzero(join_check) == np.count_nonzero(split_check):
            print("No new joins or splits have occurred")
        elif np.count_nonzero(join_check) > np.count_nonzero(split_check):
            print("Split occurred")
        else:
            print("Join occurred")

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
        file_count = len(self.file_list)
        file_counter = 1
        for f in self.file_list:
            print("Busy with file", str(file_counter), "of", str(file_count), "files")
            file_counter += 1
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
            #self._aggregate_across_samples(values_for_experts)
            first_image = self.file_list[3]
            self._image_differences_test(io.imread(first_image[0]), sample_name=first_image[1])
        if save_path is not None:
            with open(save_path + "lw_thrsh_metrics.json", 'w') as j:
                json.dump(self.low_thresholds, j)

    def stack_hist_plot(self):
        sample_storage = {"Sample":[], "intensity_range": [], "counts": []}
        sample_voxels = {"Sample":[], "Source":[], "Voxel Count":[]}
        sample_thresholds = {}
        expert_count = 0
        for f in self.file_list:
            f_path = f[0]
            f_name = f[1]
            if f_name in self.exp_threshes:
                auto_values = self.automatic_thresholds[f_name]
                inverted_mean = (auto_values[1][0][0] + auto_values[1][0][1] + auto_values[1][0][2]) / 3
                logistic_mean = (auto_values[1][1][0] + auto_values[1][1][1] + auto_values[1][1][2]) / 3
                exp_values = self.exp_threshes[f_name]
                expert_count = max(len(exp_values), expert_count)
                exp_low = min([exp_values[e_low][0] for e_low in range(len(exp_values)) if exp_values[e_low] is not None])
                low_cutoff = min(auto_values[0], exp_low)
                sample_thresholds[f_name] = {"Expert": exp_values, "Auto": [auto_values[0], inverted_mean, logistic_mean]}
                f_image = io.imread(f_path)
                intensity_counts, intensity_range = histogram(f_image, nbins=256)
                cut_off_range = np.nonzero(intensity_range >= low_cutoff)
                intensity_counts, intensity_range = intensity_counts[cut_off_range], intensity_range[cut_off_range]
                sample_thresholds[f_name]["Fill_details"] = [intensity_range, intensity_counts]
                sample_label = [f_name for n in intensity_range]
                sample_storage["Sample"] += sample_label
                sample_storage["intensity_range"] += intensity_range.tolist()
                sample_storage["counts"] += intensity_counts.tolist()

                for ex in range(len(exp_values)):
                    if exp_values[ex] is not None:
                        sample_voxels["Sample"].append(f_name)
                        sample_voxels["Source"].append("E" + str(ex))
                        exp_thresh = exp_values[ex][1] if exp_values[ex][1] > exp_values[ex][0] else exp_values[ex][1] + 0.01
                        thresholded = self._threshold_image(f_image, exp_values[ex][0], exp_thresh)
                        sample_voxels["Voxel Count"].append(thresholded.sum())
                for inv in range(len(auto_values[1][0])):
                    inv_values = auto_values[1][0][inv]
                    sample_voxels["Sample"].append(f_name)
                    sample_voxels["Source"].append("I" + str(inv))
                    thresholded = self._threshold_image(f_image, auto_values[0], inv_values)
                    sample_voxels["Voxel Count"].append(thresholded.sum())
                for logi in range(len(auto_values[1][1])):
                    logi_values = auto_values[1][1][logi]
                    sample_voxels["Sample"].append(f_name)
                    sample_voxels["Source"].append("L" + str(logi))
                    thresholded = self._threshold_image(f_image, auto_values[0], logi_values)
                    sample_voxels["Voxel Count"].append(thresholded.sum())

                # histogram_grid = sns.FacetGrid(histogram_compare_2, col="Norm", sharey=False)
                # histogram_grid.map(sns.lineplot, "intensity_range", "counts")
        sample_hist_dfs = pd.DataFrame.from_dict(sample_storage)
        sns.set_palette('bone')
        sample_hist_plots = sns.relplot(sample_hist_dfs, x="intensity_range", y="counts", col="Sample", height=3, col_wrap=5, kind='line',
                                        facet_kws={"sharey":False, "sharex":False})

        colour_range = sns.color_palette('tab10')[1:]

        def colour_line(sample_name, axes):
            last_value = 0
            colour_counter = 0
            expert_thresh = sample_thresholds[sample_name]["Expert"]
            for exp in range(len(expert_thresh)):
                expert = expert_thresh[exp]
                if expert is not None:
                    axes.axvline(expert[0], label="Expert" + str(exp) + "Low", linestyle='--', color=colour_range[exp])
                    axes.axvline(expert[1], label="Expert" + str(exp) + "High", color=colour_range[exp])
                    last_value = max(last_value, expert[0], expert[1])
                colour_counter += 1
            automatic_values = sample_thresholds[sample_name]["Auto"]
            axes.axvline(automatic_values[0], label="AutoLow", linestyle='--', color=colour_range[colour_counter])
            colour_counter += 1
            axes.axvline(automatic_values[1], label="InvertedMean", color=colour_range[colour_counter], linestyle='-.')
            colour_counter += 1
            axes.axvline(automatic_values[2], label="LogisticMean", color=colour_range[colour_counter], linestyle='-.')
            last_value = max(last_value, automatic_values[0], automatic_values[1], automatic_values[2])
            last_value = last_value + 10 if last_value + 10 < 255 else 255
            last_value = math.ceil(last_value/10)*10
            axes.set_xbound(lower=None, upper=last_value)
            handles, labels = axes.get_legend_handles_labels()
            '''fill_data = sample_thresholds[sample_name]["Fill_details"]
            within_range = np.nonzero(fill_data[0] < last_value + 1)
            fill_data = [fill_data[0][within_range], fill_data[1][within_range]]
            axes.fill_between(fill_data[0], fill_data[1])'''
            return [handles, labels]

        axes = sample_hist_plots.axes
        legend_labels = [[]]
        for ax in axes:
            sample_title = ax.get_title()
            new_title = sample_title.split('= ')[1]
            ax.set_title(new_title)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            legend_details = colour_line(new_title, ax)
            legend_labels = legend_details if len(legend_details[0]) > len(legend_labels[0]) else legend_labels
        sb_legend = {legend_labels[1][t]: legend_labels[0][t] for t in range(len(legend_labels[1]))}
        sample_hist_plots.add_legend(legend_data=sb_legend, title="Threshold Names")
        # plt.legend(legend_labels[0], legend_labels[1], loc='center right')
        plt.show()

        voxel_df = pd.DataFrame.from_dict(sample_voxels)
        voxel_bars = sns.catplot(voxel_df, x="Source", y="Voxel Count", col="Sample", col_wrap=5, aspect=1, kind="bar", height=2, sharey=False, legend=True)
        axes = voxel_bars.axes
        for ax in axes:
            sample_title = ax.get_title()
            new_title = sample_title.split('= ')[1]
            ax.set_title(new_title)
            '''ax.set_xlabel("")
            ax.set_xticks([])
            print(ax.patches[0].get_facecolor())
            print(type(ax))
            try:
                print(ax.container.get_label())
            except Exception as e:
                print('List or Tuple', e)
            try:
                print(ax.container[0].get_label())
            except Exception as e:
                print(e)'''
        plt.show()

    def compare_thresholds_between(self):
        '''
        This method will be used to graph the low and high thresholds for the automated system and the experts for all samples
        :return:
        '''
        ''' Order for expert thresholds and auto thresholds:
        Expert thresholds: {Sample:[(low,high)]} where each Sample has a corresponding list of tuples where each tuple is a different expert. Order for experts
        is maintained across samples
        Auto thresholds: {Sample:[low, [[invert_high_1, invert_high_2, invert_high_3], [logistic_high_1, logistic_high_2, logistic_high_3]]] where each sample
        has a list that is specifically ordered. The first element is the low threshold and the second element is an order list of nested lists. For the 
        first nesting layer the first and second elements are for the inverted and logistic high thresholds respectively. Within these inverted and logistic
        elements are lists of length three that have been nested. Element 0 of the nested list are with weighting option 0, element 1 is for weighting option 1
        and likewise for element 2 for weighting option 2.
        '''
        organised_data = {"Sample": [], "Source": [], "ThreshType": [], "ThreshValue": []}
        scatter_data = {"Sample": [], "Source": [], "High": [], "Low": [], "Row":[]}
        mean_auto_data = {"Sample": [], "Source": [], "High": [], "Low": []}

        row_separation = 5

        def expert_tracking(expert_index):
            return "Exp" + str(expert_index)
        sample_count = 0
        row_label = 1
        for samples in list(self.exp_threshes):
            sample_count += 1
            if sample_count > row_separation:
                row_label += 1
                sample_count = 1
            expert_values = self.exp_threshes[samples]
            auto_values = self.automatic_thresholds[samples]
            for et in range(len(expert_values)):
                if expert_values[et] is not None:
                    expert_results = expert_values[et]
                    organised_data["Sample"].append(samples)  # for high and low threshold
                    organised_data["Sample"].append(samples)
                    organised_data["ThreshType"].append("ExpLow")
                    organised_data["ThreshValue"].append(expert_results[0])
                    organised_data["ThreshType"].append("ExpHigh")
                    organised_data["ThreshValue"].append(expert_results[1])
                    organised_data["Source"].append(expert_tracking(et))  # expert name for each threshold type (low & high)
                    organised_data["Source"].append(expert_tracking(et))
                    #  data to be used for scatterplots
                    scatter_data["Sample"].append(samples)
                    scatter_data["Source"].append(expert_tracking(et))
                    scatter_data["Low"].append(expert_results[0])
                    scatter_data["High"].append(expert_results[1])
                    scatter_data["Row"].append(row_label)
                    # data to be used for scatterplots but the automatic high thresholds have been averaged for easier reading
                    mean_auto_data["Sample"].append(samples)
                    mean_auto_data["Source"].append(expert_tracking(et))
                    mean_auto_data["Low"].append(expert_results[0])
                    mean_auto_data["High"].append(expert_results[1])

            # These 6 are for the 6 automated results
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Inv0")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][0][0])
            organised_data["Source"].append("Inv0")

            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Inv1")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][0][1])
            organised_data["Source"].append("Inv1")

            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Inv2")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][0][2])
            organised_data["Source"].append("Inv2")

            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Log0")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][1][0])
            organised_data["Source"].append("Log0")

            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Log1")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][1][1])
            organised_data["Source"].append("Log1")

            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoLow")
            organised_data["ThreshValue"].append(auto_values[0])
            organised_data["Source"].append("Log2")
            organised_data["Sample"].append(samples)
            organised_data["ThreshType"].append("AutoHigh")
            organised_data["ThreshValue"].append(auto_values[1][1][2])
            organised_data["Source"].append("Log2")
            # The automatic scatter data
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Inv0")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][0][0])
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Inv1")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][0][1])
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Inv2")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][0][2])
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Log0")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][1][0])
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Log1")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][1][1])
            scatter_data["Sample"].append(samples)
            scatter_data["Source"].append("Log2")
            scatter_data["Low"].append(auto_values[0])
            scatter_data["High"].append(auto_values[1][1][2])
            scatter_data["Row"].append(row_label)
            scatter_data["Row"].append(row_label)
            scatter_data["Row"].append(row_label)
            scatter_data["Row"].append(row_label)
            scatter_data["Row"].append(row_label)
            scatter_data["Row"].append(row_label)
            # the below is for the Inverted mean
            mean_auto_data["Sample"].append(samples)
            mean_auto_data["Source"].append("Inverted")
            mean_auto_data["Low"].append(auto_values[0])
            auto_mean_value = (auto_values[1][0][0] + auto_values[1][0][1] + auto_values[1][0][2]) / 3
            mean_auto_data["High"].append(auto_mean_value)
            # the below is for the Logistic mean
            mean_auto_data["Sample"].append(samples)
            mean_auto_data["Source"].append("Logistic")
            mean_auto_data["Low"].append(auto_values[0])
            auto_mean_value = (auto_values[1][1][0] + auto_values[1][1][1] + auto_values[1][1][2]) / 3
            mean_auto_data["High"].append(auto_mean_value)

        scatter_df = pd.DataFrame.from_dict(scatter_data)
        organised_df = pd.DataFrame.from_dict(organised_data)
        mean_auto_df = pd.DataFrame.from_dict(mean_auto_data)
        self._convert_to_graph(organised_df, mean_auto_df)

    def _convert_to_graph(self, organised, scatter):
        '''
        This method exists to partition the compare_thresholds_between function for debugging the plots separately to the data
        :param organised:
        :param scatter:
        :return:
        '''
        sns.set_theme()
        sns.set(font_scale=0.8)
        scatterplot = sns.relplot(data=scatter, x="Low", y="High", col="Sample", hue="Source", col_wrap=5, aspect=1, height=2, legend="auto")
        plt.show()
        barplots = sns.catplot(data=organised, x="Source", y="ThreshValue", hue="ThreshType", col="Sample", col_wrap=5, aspect=1, height=2, kind="bar")
        '''sample_grid = sns.FacetGrid(organised, col="Sample", col_wrap=5, aspect=1, height=2, legend_out=True)
        sample_grid.map(sns.barplot, "Source", "ThreshValue", "ThreshType", **{"palette":"deep"})'''
        plt.show()

    def low_threshold_similarity(self, low_thresh_location):
        expert_thresh_file = "C:\\RESEARCH\\Mitophagy_data\\gui params\\rensu_thresholds.json"
        expert_thresholds = {}
        auto_lows = {}
        distance_results = {}
        with open(expert_thresh_file, "r") as j:
            expert_thresholds = json.load(j)
        with open(low_thresh_location, "r") as j:
            auto_lows_data = json.load(j)
        for f in self.file_list:
            distance_results[f[1]] = {}
            image = io.imread(f[0])
            time_set = self._prepare_image(image, f[1])
            expert_low_and_high = expert_thresholds[f[1]]
            for t in range(0, len(time_set)):
                img = time_set[t]
                auto_low = auto_lows_data[f[1] + " " + str(t)]
                low_values_to_calc = {"Normal":auto_low["Normal"], "Log":auto_low["Log"],
                                      "Otsu":auto_low["Otsu"], "Triangle":auto_low["Triangle"]}
                distance_results[f[1]][t] = self._low_thresh_sim_calc_disc(img, expert_low_and_high["low"],
                                                                       low_values_to_calc, expert_low_and_high["high"])
                '''with open("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\System Metrics\\autoDist.json", "w") as j:
                     json.dump(distance_results, j)'''
        for k, v in distance_results.items():
            print(k, v)



    def _low_thresh_sim_calc_disc(self, image, expert_low, auto_lows, high_thresh):
        '''auto_min, auto_max = min(auto_lows), max(auto_lows)
        range_min = min(expert_low, auto_min)
        range_max = max(expert_low, auto_max)
        res_range = range_max - range_min'''
        target_image = self._threshold_image(image, int(expert_low), int(high_thresh))
        target_labels, _unneeded = ndi.label(target_image)
        auto_distances = {}
        for auto_label, low_threshold in auto_lows.items():
            varied_labels, _unneeded = ndi.label(self._threshold_image(image, int(low_threshold), int(high_thresh)))
            distance, penalty = self._distance_from_target(varied_labels, target_labels)

            if distance == 0:
                print("Zero overlap?")
            if penalty >= distance:
                print("Bad penalty!!!!!!!!!!!!!!!!!!!!!")
            auto_distances[auto_label] = [str(distance), str(penalty), str(distance - penalty)]
        return auto_distances

    def save_histogram(self, image_path, cutoff=None):
        intensity_counts, intensities = histogram(io.imread(image_path))
        thesis_figs = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Thesis Figures\\"
        plt.plot(intensities, intensity_counts)
        plt.axvline(x=cutoff, c='k')
        plt.savefig(fname=thesis_figs + "hist.png")
        plt.clf()
        plt.cla()
        log_counts = np.log1p(intensity_counts)
        plt.plot(intensities, log_counts)
        plt.axvline(x=cutoff, c='k')
        plt.savefig(fname=thesis_figs + "hist_log.png")
        plt.clf()
        plt.cla()
        if cutoff is not None:
            valid_values = np.greater_equal(intensities, cutoff)
            intensity_counts, intensities = intensity_counts[valid_values], intensities[valid_values]
            log_counts = log_counts[valid_values]
        plt.plot(intensities, intensity_counts)
        plt.axvline(x=cutoff, c='k')
        plt.savefig(fname=thesis_figs + "hist_cut.png")
        plt.clf()
        plt.cla()
        plt.plot(intensities, log_counts)
        plt.axvline(x=cutoff, c='k')
        plt.savefig(fname=thesis_figs + "hist_log_cut.png")
        plt.clf()
        plt.cla()

    def generate_threshold_graphs(self):
        thresh_metrics_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\System_metrics\\"
        auto_distances = {}
        auto_lows = {}
        df_dict = {"Sample":[], "Threshold Option":[], "Distance":[]}
        with open(thresh_metrics_path + "autoDist.json", "r") as j:
            for sample, dists in json.load(j).items():
                actual_dists = dists["0"]
                auto_distances[sample] = {"Normal":actual_dists["Normal"][2], "Log":actual_dists["Log"][2], "Otsu":actual_dists["Otsu"][2],
                                          "Triangle":actual_dists["Triangle"][2]}
        with open(thresh_metrics_path + "lw_thrsh_metrics.json", "r") as j:
            for sample, thresholds in json.load(j).items():
                chosen_thresh = "Normal" if thresholds["Normal"] >= thresholds["Otsu"] else "Log"
                auto_lows[sample.split(" ")[0]] = {"Normal":thresholds["Normal"], "Log":thresholds["Log"], "Otsu":thresholds["Otsu"], "Triangle":thresholds["Triangle"],
                                     "Chosen":chosen_thresh}

        def box_setup():
            for k, v in auto_lows.items():
                dist_values = auto_distances[k]
                dist_values["Knee"] = dist_values[v["Chosen"]]
                for t in ["Otsu", "Triangle", "Knee"]:
                    df_dict["Sample"].append(k)
                    df_dict["Threshold Option"].append(t)
                    df_dict["Distance"].append(float(dist_values[t]))

        box_setup()
        low_thresh_df = pd.DataFrame.from_dict(df_dict)
        print(low_thresh_df)
        sns.boxplot(data=low_thresh_df, y="Threshold Option", x="Distance", order=["Knee", "Otsu", "Triangle"])
        plt.show()
        thresh_option_mean = low_thresh_df[["Threshold Option", "Distance"]].groupby("Threshold Option").mean()
        sns.barplot(data=thresh_option_mean, x=thresh_option_mean.index, y="Distance")
        plt.show()


    def generate_ihh_figures(self, im_path, low_thresh, sample_name = None, save_location = None):
        if type(im_path) is str:
            image = io.imread(im_path)
        else:
            image = im_path
        intensities, threshold_counts = self._efficient_hysteresis_iterative(image, low_thresh)
        intensities, threshold_counts = intensities[:-1], threshold_counts[:-1]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        if sample_name is not None:
            fig.suptitle(sample_name)
        sns.lineplot(x=intensities, y=threshold_counts, ax=ax1)
        ax1.set_title("Original IHH")
        ax1.set_xlabel("High Threshold Intensities")
        ax1.set_ylabel("Non-zero voxel count")
        slopes, slope_points = self._get_slope(intensities, threshold_counts)
        mving_slopes = self._moving_average(slopes, window_size=8)
        sns.lineplot(x=slope_points, y=slopes, ax=ax2)
        ax2.set_title("IHH Slope")
        ax2.set_xlabel("High Threshold Intensities")
        ax2.set_ylabel("Relative change in count")
        sns.lineplot(x=slope_points, y=mving_slopes, ax=ax3)
        ax3.set_title("Smoothed IHH Slope")
        ax3.set_xlabel("High Threshold Intensities")
        ax3.set_ylabel("Relative change in count")
        if save_location is None:
            plt.show()
        else:
            full_save_path = save_location + sample_name if sample_name is not None else save_location
            ax1.figure.set_size_inches(8, 6)
            ax2.figure.set_size_inches(8, 6)
            ax3.figure.set_size_inches(8, 6)
            plt.tight_layout()
            plt.savefig(full_save_path, dpi=100)
            plt.clf()
            plt.cla()

    def go_through_image_ihh(self, save_location=None):
        for f in self.file_list:
            print("Image currently", f[1])
            image = self._grayscale(io.imread(f[0]))
            lw_thrsh = self._low_select(img=image)[0]
            start_time1 = time.process_time()
            # intens, thresh1 = self._efficient_hysteresis_iterative(image, lw_thrsh)
            # print("Threshold 1 values", thresh1)
            end_time1 = time.process_time()
            '''plt.plot(intens, thresh1)
            plt.show()'''
            # print(len(thresh1), len(intens))
            print("Next")
            start_time2 = time.process_time()
            intens, thresh2 = self._efficient_hysteresis_iterative_time(image, lw_thrsh, 2.2, True)
            end_time2 = time.process_time()
            print("Time taken for old", end_time1 - start_time1, "for new", end_time2 - start_time2)
            # plt.plot(intens, thresh2)
            # plt.show()
            print("Threshold results")
            # print(thresh1)
            # print(thresh2)
            # self.generate_ihh_figures(image, lw_thrsh, f[1], save_location=save_location)

    def generate_IHH_plots(self, ihh_sample):
        '''will generate IHH figures for the high threshold generation but will also store the line plot details in
        json file for future editing and annotation. Plots needed are the IHH graph, the slope version, the rescaling
        distribution, the effect of the rescaling distribution (This could all be done with a seaborn jointgrid?).
        In future the centroid plots and window effects will need to be plotted.'''
        sample_image = io.imread(self.image_paths[ihh_sample])
        low_thr, validity = self._low_select(sample_image)
        intensities, threshold_counts, indep_count = self._efficient_hysteresis_iterative(sample_image, low_thr, True)
        print("Ind structure count", indep_count)
        intensities, threshold_counts = intensities[:-1], threshold_counts[:-1]
        slopes, slope_points = self._get_slope(intensities, threshold_counts)
        mving_slopes = self._moving_average(slopes, window_size=8)
        max_slope = math.ceil(max(slopes))
        '''inverted_rescaler = np.arange(max_slope, 0, -1)/max_slope
        print(inverted_rescaler.shape, len(mving_slopes))
        logist = self._generate_sigmoid(max_slope / 2, k=6)
        print(logist)
        logist_rescaled = np.array([logist[int(lgr)] for lgr in mving_slopes])
        logist = np.array(logist)
        print(logist.shape, logist_rescaled.shape, max_slope)
        reweighted_dist = self._apply_weights(logist_rescaled, mving_slopes)
        print("Reweighted length", len(reweighted_dist))'''
        '''sns.lineplot(x=np.arange(len(logist)), y=logist)
        plt.show()
        sns.lineplot(x=np.arange(len(inverted_rescaler)), y=inverted_rescaler)
        plt.show()'''
        '''print(len(slope_points), len(reweighted_dist), len(inverted_rescaler), len(mving_slopes))
        new_resolution = math.ceil(len(logist)/len(reweighted_dist))
        print(new_resolution)
        new_range = [logist[int(lgr * new_resolution)] for lgr in range(len(reweighted_dist))]
        print(len(new_range))
        print(max(logist_rescaled))
        g = sns.JointGrid()
        x = slope_points
        sns.lineplot(x=x, y=(np.array(mving_slopes)/max(mving_slopes))*225, ax=g.ax_joint)
        sns.lineplot(x=x, y=logist_rescaled/logist_rescaled.max(), ax=g.ax_marg_x)
        sns.lineplot(x=new_range, y=np.arange(0, len(new_range), step=1), ax=g.ax_marg_y)
        plt.show()'''
        voxel_arr = np.array(threshold_counts)
        high_thresh0 = self._logistic_thresholding(mving_slopes, np.power(voxel_arr, 1), steepness=50, weighted_option=0) + low_thr
        print("High thresh ver 0", high_thresh0)
        high_thresh1 = self._logistic_thresholding(mving_slopes, np.power(voxel_arr, 1), steepness=50, weighted_option=1) + low_thr
        print("High thresh ver 1", high_thresh1)
        high_thresh2 = self._logistic_thresholding(mving_slopes, np.power(voxel_arr, 1), steepness=50, weighted_option=2) + low_thr
        print("High thresh ver 2", high_thresh2)
        sns.lineplot(x=intensities, y=threshold_counts)
        plt.axvline(x=high_thresh0, c='r')
        plt.axvline(x=high_thresh1, c='g')
        plt.axvline(x=high_thresh2, c='b')
        plt.show()
        high_thresh_test = self._logistic_thresholding(mving_slopes, threshold_counts, steepness=50, weighted_option=3)
        print("Efficient test", high_thresh_test)

    def extract_information_samples(self):
        thresholding_information = {}
        structure_count_info = {}
        save_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\"
        sample_len = len(self.file_list)
        sample_counter = 1
        for f in self.file_list:
            print("Sample", str(sample_counter), " of", str(sample_len))
            sample_counter += 1
            image = io.imread(f[0])
            file_name = f[1]
            print("Sample", file_name)
            thresholding_information[file_name] = {}
            low_thr, validity = self._low_select(image)
            thresholding_information[file_name]["low_thr"] = str(low_thr)
            intensities, threshold_counts = self._efficient_hysteresis_iterative(image, low_thr, False)
            intensities, threshold_counts = intensities[:-1], threshold_counts[:-1]
            slopes, slope_points = self._get_slope(intensities, threshold_counts)
            mving_slopes = self._moving_average(slopes, window_size=8)
            # structure_count_info[file_name] = indep_count
            '''with open(save_path + "struct_count_ihh.json", 'w') as j:
                json.dump(structure_count_info, j)'''
            # thresholding_information[file_name]["Indep Count"] = indep_count
            voxel_arr = np.array(threshold_counts)
            for p in [1, 0.5, 0.3]:
                high_thresh_11 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=6, weighted_option=0) + low_thr)
                high_thresh_12 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=6, weighted_option=1) + low_thr)
                high_thresh_13 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=6, weighted_option=2) + low_thr)
                high_thresh_21 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=50, weighted_option=0) + low_thr)
                high_thresh_22 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=50, weighted_option=1) + low_thr)
                high_thresh_23 = str(self._logistic_thresholding(mving_slopes, np.power(voxel_arr, p), steepness=50, weighted_option=2) + low_thr)
                thresholding_information[file_name]["Voxel pow" + str(p)] = [[high_thresh_11, high_thresh_12, high_thresh_13],
                                                                             [high_thresh_21, high_thresh_22, high_thresh_23]]
                with open(save_path + "power_high_thresh.json", 'w') as j:
                    json.dump(thresholding_information, j)

    def visualise_struct_counts(self):
        json_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\struct_count_ihh.json"
        struct_count_values = None
        with open(json_path, "r") as j:
            struct_count_values = json.load(j)
        for sample, values in struct_count_values.items():
            values.reverse()
            slopes, slope_points = self._get_slope(x=list(range(len(values))), y=values)
            print(slopes)
            mving_slopes = self._moving_average(slopes, window_size=3)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            norm_val = np.array(values) / max(values)
            norm_slopes = np.array(slopes) / max(slopes)
            norm_mving = np.array(mving_slopes) / max(mving_slopes)
            sns.lineplot(x=np.arange(len(values)), y=norm_val, ax=ax1)
            sns.lineplot(x=slope_points, y=norm_slopes, ax=ax2)
            sns.lineplot(x=slope_points, y=norm_mving, ax=ax3)
            fig.suptitle(sample)
            plt.show()


    def get_spatial_metrics_more(self):
        structure_count_info = {}
        save_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\"
        sample_len = len(self.file_list)
        sample_counter = 1
        for f in self.file_list:
            print("Sample", str(sample_counter), " of", str(sample_len))
            sample_counter += 1
            image = io.imread(f[0])
            file_name = f[1]
            print("Sample", file_name)
            structure_count_info[file_name] = {}
            low_thr, validity = self._low_select(image)
            structure_count_info[file_name]["low_thr"] = str(low_thr)
            intensities, threshold_counts, structure_sizes = self._efficient_hysteresis_iterative(image, low_thr, False, True)
            intensities, threshold_counts = intensities[:-1], threshold_counts[:-1]
            slopes, slope_points = self._get_slope(intensities, threshold_counts)
            mving_slopes = self._moving_average(slopes, window_size=8)
            print(type(structure_sizes), type(threshold_counts[0]), type(mving_slopes[0]))
            structure_count_info[file_name]["Structures Lost"] = structure_sizes
            structure_count_info[file_name]["ihh"] = threshold_counts
            structure_count_info[file_name]["slopes"] = mving_slopes
            with open(save_path + "spatio_struct_info.json", 'w') as j:
                json.dump(structure_count_info, j)
            print("Stored")

    def visualise_spatial_info(self):
        spatial_info_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\spatio_struct_info.json"
        with open(spatial_info_path, "r") as j:
            spatial_info = json.load(j)
        for sample, values in spatial_info.items():
            print(sample)
            low_thrsh = values["low_thr"]
            ihh_data = values["ihh"]
            ihh_data.reverse()
            norm_ihh = (np.array(ihh_data)/max(ihh_data))[:-1]
            slope_data = values["slopes"]
            struct_info = values["Structures Lost"]
            structures_present = []
            current_total_mean = []
            current_structures = None
            prior_structures = None
            struct_size_change = []
            struct_size_past = None
            struct_size_gradient = []
            struct_size_gradient_prior = None
            struct_count = []
            added_struct_count = []
            average_struct_size_present = []
            voxels_added = []
            for g in range(len(struct_info) - 1, 0, -1):
                # print(current_structures, prior_structures, len(struct_info[g]))
                if current_structures is None and len(struct_info[g]) > 0:
                    current_structures = np.array(struct_info[g])
                    prior_structures = np.array(struct_info[g])
                    struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient.append(struct_size_gradient_prior)
                elif len(struct_info[g]) == 0 and struct_size_past is not None:
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient.append(struct_size_gradient_prior)
                elif current_structures is not None and prior_structures is not None and struct_size_past is not None and len(struct_info[g]) > 0:
                    current_structures = np.array(struct_info[g])
                    struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean()) + struct_size_past
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_gradient.append(struct_size_gradient_prior)
                    prior_structures = np.array(struct_info[g])
                else:
                    pass
                if current_structures is not None:
                    structures_present += struct_info[g]
                    struct_count.append(len(structures_present))
                    added_struct_count.append(len(struct_info[g]))
                    current_total_mean.append(np.array(structures_present).mean())
                    struct_sze = np.array(struct_info[g]).mean() if len(struct_info[g]) > 0 else 0
                    average_struct_size_present.append(struct_sze)
                    voxels_added.append(sum(struct_info[g]))
            struct_size_change.reverse()
            struct_size_gradient.reverse()
            current_total_mean.reverse()
            added_struct_count.reverse()
            struct_count.reverse()
            average_struct_size_present.reverse()
            voxels_added.reverse()
            cum_change_mean = np.array(struct_size_change).mean()
            # leftmost_intersect_intensity = np.array(np.where(abs(struct_size_change-cum_change_mean) == np.amin(abs(struct_size_change-cum_change_mean))))
            leftmost_intersect_intensity = []
            for t in range(1, len(struct_size_change)): #need to interpolate to determine actual intersections. go from left to right
                if struct_size_change[t - 1] < cum_change_mean < struct_size_change[t]:
                    if abs(struct_size_change[t - 1] - cum_change_mean) < abs(struct_size_change[t] - cum_change_mean):
                        leftmost_intersect_intensity.append(t - 1)
                    else:
                        leftmost_intersect_intensity.append(t)
                else:
                    if cum_change_mean == struct_size_change[t - 1]:
                        leftmost_intersect_intensity.append(t - 1)
                    if cum_change_mean == struct_size_change[t]:
                        leftmost_intersect_intensity.append(t)
            leftmost_intersect_intensity = min(leftmost_intersect_intensity)
            print(len(struct_size_change), len(struct_size_gradient))
            left_peak_change = np.where(struct_size_gradient[0:leftmost_intersect_intensity+1]==np.amin(struct_size_gradient[0:leftmost_intersect_intensity+1]))
            left_peak_change = int(min(left_peak_change)[0])
            print("Intersection point", left_peak_change + int(low_thrsh))

            def density_calc(input_values):
                max_dist = len(input_values) - 1
                k = 0.1
                input_array = np.array(input_values)
                index_range = np.arange(len(input_values))
                density_weights = np.zeros(tuple([len(input_values), len(input_values)]))
                for r in range(len(input_values)):
                    distances = np.abs(index_range - r)
                    # offset = (len(input_values) - r)/len(input_values)
                    density_weights[r] = np.exp(-1 * k * distances)
                    # density_weights[r][:r] = 1 / (1 + np.exp(-1 * k * ((max_dist - distances[:r]) + max_dist * offset)))
                density_weighted_values = density_weights * input_array
                density_totals = np.sum(density_weighted_values, axis=1)
                '''figalphs, (ax_aleph, ax_elaph, ax_sigrun) = plt.subplots(3, 1)
                sns.lineplot(y=input_array, x=index_range, ax=ax_aleph)
                sns.lineplot(y=density_weighted_values[0], x=index_range, ax=ax_elaph)
                sns.lineplot(y=density_weights[101], x=index_range, ax=ax_sigrun)'''
                return density_totals

            def triangle_centroid_test():
                triangle_range = np.arange(15, -1, -1)
                linear_range = np.arange(16)
                print(triangle_range)
                centroid = np.trapz(y=triangle_range * linear_range) / np.trapz(triangle_range)
                sns.lineplot(y=triangle_range, x=linear_range)
                plt.axvline(x=centroid)
                plt.show()
            # triangle_centroid_test()

            def flip_struct_count(count_to_flip, flip=True):
                count_array = np.array(count_to_flip)
                if flip:
                    count_array = np.absolute(count_array - count_array.max())
                return count_array/count_array.max()

            def apply_flip_to_ihh():
                graph_array = np.array(struct_size_change)
                difference_array = np.zeros_like(graph_array)
                difference_array[left_peak_change:] = np.absolute(graph_array[left_peak_change:] - graph_array[left_peak_change])
                graph_array = graph_array - 2 * difference_array
                norm_graph = graph_array - np.amin(graph_array)
                norm_graph = norm_graph / norm_graph.max()
                adjusted_ihh = norm_ihh * norm_graph
                adjusted_ihh /= adjusted_ihh.max()
                reweighted_norm_graph = norm_graph * flip_struct_count(current_total_mean, False) * norm_ihh
                reweighted_norm_graph /= reweighted_norm_graph.max()
                fig_ihh, (ax_one, ax_two, ax_three, ax_four) = plt.subplots(4, 1)
                sns.lineplot(y=norm_ihh, x=np.arange(int(low_thrsh), int(low_thrsh) + len(adjusted_ihh)), ax=ax_one)
                sns.lineplot(y=np.power(norm_ihh, 0.5), x=np.arange(int(low_thrsh), int(low_thrsh) + len(adjusted_ihh)), ax=ax_two)
                sns.lineplot(y=adjusted_ihh, x=np.arange(int(low_thrsh), int(low_thrsh) + len(adjusted_ihh)), ax=ax_three)
                sns.lineplot(y=reweighted_norm_graph, x=np.arange(int(low_thrsh), int(low_thrsh) + len(adjusted_ihh)), ax=ax_four)

            # apply_flip_to_ihh()

            def flip_grads_around_intersect():
                graph_array = np.array(struct_size_change)
                difference_array = np.zeros_like(graph_array)
                difference_array[left_peak_change:] = np.absolute(graph_array[left_peak_change:] - graph_array[left_peak_change])
                graph_array = graph_array - 2 * difference_array
                norm_graph = graph_array - np.amin(graph_array)
                norm_graph = norm_graph / norm_graph.max()
                # density_calc(norm_graph)
                norm_range = np.arange(len(struct_size_change))/(len(struct_size_change) - 1)
                scaled_range = (np.arange(len(struct_size_change)) * norm_graph)
                centr1 = (scaled_range * norm_range).sum()/scaled_range.sum() * (len(struct_size_change) - 1)
                centr2 = ((scaled_range * flip_struct_count(current_total_mean, False)) * norm_range).sum()/(scaled_range * flip_struct_count(current_total_mean, False)).sum() * (len(struct_size_change) - 1)
                test_centr_1 = np.trapz(y=scaled_range * norm_range)/np.trapz(scaled_range)
                test_centr_2 = np.trapz(y=scaled_range * flip_struct_count(current_total_mean, False) * norm_range) / np.trapz(scaled_range * flip_struct_count(current_total_mean, False))
                print(centr1, centr2)
                print("Centroid by integral", test_centr_1 * (len(struct_size_change) - 1), test_centr_2 * (len(struct_size_change) - 1))
                figtemp, (axa, axb, axc) = plt.subplots(3, 1)
                sns.lineplot(y=struct_size_change, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=axa)
                sns.lineplot(y=norm_graph, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=axb)
                axa.axvline(x=left_peak_change + int(low_thrsh), c='g')
                axb.axvline(x=left_peak_change + int(low_thrsh), c='g')
                axb.axvline(x=centr1 + int(low_thrsh), c='k')
                axb.axvline(x=centr2 + int(low_thrsh), c='r')
                reweighted_norm_graph = norm_graph * flip_struct_count(current_total_mean, False)
                sns.lineplot(y=reweighted_norm_graph/reweighted_norm_graph.max(), x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=axc)
                axc.axvline(x=centr2 + int(low_thrsh), c='r')

            # flip_grads_around_intersect()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            sns.lineplot(y=struct_size_change, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=ax1)
            ax1.set_title("Cumulative change in mean struct size descending")
            ax1.axhline(y=cum_change_mean, c='r')
            ax1.axvline(x=left_peak_change+int(low_thrsh), c='g')
            sns.lineplot(y=struct_size_gradient, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_gradient)), ax=ax2)
            ax2.set_title("Relative change in average struct size between intensities descending")
            ax2.axhline(y=np.array(struct_size_gradient).mean(), c='r')
            ax2.axvline(x=left_peak_change + int(low_thrsh), c='g')
            sns.lineplot(y=current_total_mean, x=np.arange(int(low_thrsh), int(low_thrsh) + len(current_total_mean)), ax=ax3)
            ax3.set_title("Mean structure size of all structures currently present")

            fig2, (ax4, ax5, ax6) = plt.subplots(3, 1)
            sns.lineplot(y=average_struct_size_present, x=np.arange(int(low_thrsh), int(low_thrsh) + len(average_struct_size_present)), ax=ax4)
            ax4.set_title("Average Structure size of newly added structures")
            sns.lineplot(y=voxels_added, x=np.arange(int(low_thrsh), int(low_thrsh) + len(voxels_added)), ax=ax5)
            ax5.set_title("Total number of independent structures")
            sns.lineplot(y=density_calc(voxels_added), x=np.arange(int(low_thrsh), int(low_thrsh) + len(voxels_added)), ax=ax6)
            ax6.set_title("Total voxels by average size")
            plt.show()


    def recalc_thresh_with_vox_weight_new(self):
        spatial_info_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\spatio_struct_info.json"
        with open(spatial_info_path, "r") as j:
            spatial_info = json.load(j)
        for sample, values in spatial_info.items():
            print(sample)
            low_thrsh = int(values["low_thr"])
            ihh_data = values["ihh"]
            ihh_data.reverse()
            norm_ihh = (np.array(ihh_data) / max(ihh_data))[:-1]
            slope_data = values["slopes"]
            slope_data.reverse()
            struct_info = values["Structures Lost"]

            structures_present = []
            current_total_mean = []
            current_structures = None
            prior_structures = None
            struct_size_change = []
            struct_size_past = None
            struct_size_gradient = []
            struct_size_gradient_prior = None
            struct_count = []
            added_struct_count = []
            average_struct_size_present = []
            for g in range(len(struct_info) - 1, 0, -1):
                if current_structures is None and len(struct_info[g]) > 0:
                    current_structures = np.array(struct_info[g])
                    prior_structures = np.array(struct_info[g])
                    struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient.append(struct_size_gradient_prior)
                elif len(struct_info[g]) == 0 and struct_size_past is not None:
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient.append(struct_size_gradient_prior)
                elif current_structures is not None and prior_structures is not None and struct_size_past is not None and len(struct_info[g]) > 0:
                    current_structures = np.array(struct_info[g])
                    struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean()) + struct_size_past
                    struct_size_change.append(struct_size_past)
                    struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                    struct_size_gradient.append(struct_size_gradient_prior)
                    prior_structures = np.array(struct_info[g])
                else:
                    pass
                if current_structures is not None:
                    structures_present += struct_info[g]
                    struct_count.append(len(structures_present))
                    added_struct_count.append(len(struct_info[g]))
                    current_total_mean.append(np.array(structures_present).mean())
                    struct_sze = np.array(struct_info[g]).mean() if len(struct_info[g]) > 0 else 0
                    average_struct_size_present.append(struct_sze)

            struct_size_change.reverse()
            struct_size_gradient.reverse()
            current_total_mean.reverse()
            added_struct_count.reverse()
            struct_count.reverse()
            average_struct_size_present.reverse()
            cum_change_mean = np.array(struct_size_change).mean()
            leftmost_intersect_intensity = []
            for t in range(1, len(struct_size_change)):
                if struct_size_change[t - 1] < cum_change_mean < struct_size_change[t]:
                    if abs(struct_size_change[t - 1] - cum_change_mean) < abs(struct_size_change[t] - cum_change_mean):
                        leftmost_intersect_intensity.append(t - 1)
                    else:
                        leftmost_intersect_intensity.append(t)
                else:
                    if cum_change_mean == struct_size_change[t - 1]:
                        leftmost_intersect_intensity.append(t - 1)
                    if cum_change_mean == struct_size_change[t]:
                        leftmost_intersect_intensity.append(t)
            leftmost_intersect_intensity = min(leftmost_intersect_intensity)
            left_peak_change = np.where(
                struct_size_gradient[0:leftmost_intersect_intensity + 1] == np.amin(struct_size_gradient[0:leftmost_intersect_intensity + 1]))
            left_peak_change = int(min(left_peak_change)[0])
            print("Point of flipping", left_peak_change + low_thrsh)
            def flip_struct_count(count_to_flip, flip=True):
                count_array = np.array(count_to_flip)
                if flip:
                    count_array = np.absolute(count_array - count_array.max())
                return count_array/count_array.max()

            def apply_flip_to_ihh():
                graph_array = np.array(struct_size_change)
                difference_array = np.zeros_like(graph_array)
                difference_array[left_peak_change:] = np.absolute(graph_array[left_peak_change:] - graph_array[left_peak_change])
                graph_array = graph_array - 2 * difference_array
                norm_graph = graph_array - np.amin(graph_array)
                norm_graph = norm_graph / norm_graph.max()
                adjusted_ihh = norm_ihh * norm_graph
                adjusted_ihh /= adjusted_ihh.max()
                reweighted_norm_graph = norm_graph * flip_struct_count(current_total_mean, False) * norm_ihh
                reweighted_norm_graph /= reweighted_norm_graph.max()
                return [norm_ihh, adjusted_ihh, reweighted_norm_graph]

            voxel_weights = apply_flip_to_ihh()

            counter = 0
            for voxel_arr in voxel_weights:
                if counter == 0:
                    print("There is no adjustment")
                    counter = 1
                elif counter == 1:
                    print("This is for the normal adjusted")
                    counter = 2
                else:
                    counter = 0
                    print("This is for the extra adjusted")
                high_thresh1 = self._logistic_thresholding(slope_data, voxel_arr, steepness=6, weighted_option=0) + low_thrsh
                high_thresh2 = self._logistic_thresholding(slope_data, voxel_arr, steepness=6, weighted_option=1) + low_thrsh
                high_thresh3 = self._logistic_thresholding(slope_data, voxel_arr, steepness=6, weighted_option=2) + low_thrsh

                print("Option 1", high_thresh1, "Option 2", high_thresh2, "Option 3", high_thresh3)
                print("Slope of 50")
                high_thresh1 = self._logistic_thresholding(slope_data, voxel_arr, steepness=50, weighted_option=0) + low_thrsh
                high_thresh2 = self._logistic_thresholding(slope_data, voxel_arr, steepness=50, weighted_option=1) + low_thrsh
                high_thresh3 = self._logistic_thresholding(slope_data, voxel_arr, steepness=50, weighted_option=2) + low_thrsh

                print("Option 1", high_thresh1, "Option 2", high_thresh2, "Option 3", high_thresh3)
                print("Average centroid", (high_thresh1 + high_thresh2 + high_thresh3)/3)
                print("Final composite", (high_thresh1 + high_thresh3 + left_peak_change + low_thrsh)/3)
                print("Compounded average", ((high_thresh1 + high_thresh3)/2 + left_peak_change + low_thrsh)/2)
                sns.lineplot(y=voxel_arr, x=np.arange(low_thrsh, len(voxel_arr) + low_thrsh))
                plt.axvline(x=left_peak_change+low_thrsh, c='k')
                plt.axvline(x=high_thresh1, c='r')
                plt.axvline(x=high_thresh2, c='g')
                plt.axvline(x=high_thresh3, c='b')
                plt.axvline(x=(high_thresh1 + high_thresh2 + high_thresh3)/3, c='y')
                plt.show()

            print("")

    def get_sample_ihh(self):
        save_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\"
        sample_data = {}
        for f in self.file_list:
            print("Image currently", f[1])
            image = self._grayscale(io.imread(f[0]))
            lw_thrsh = self._low_select(img=image)[0]
            intens, thresh = self._efficient_hysteresis_iterative(image, lw_thrsh)
            sample_data[f[1]] = {"x":intens, "y":thresh}
            with open(save_path + "sample_ihh_values.json", 'w') as j:
                json.dump(sample_data, j)

    def generate_ihh_figures(self):
        save_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\"
        ihh_data = {}
        with open(save_path + "sample_ihh_values.json", 'r') as j:
            ihh_data = json.load(j)
        for sample_name, ihh in ihh_data.items():
            sns.lineplot(x=ihh['x'], y=ihh['y'])
            plt.title(sample_name)
            plt.show()

    def place_expert_lines(self, sample_to_use):
        save_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\"
        expert_path = "C:\\RESEARCH\\Mitophagy_data\\gui params\\"
        ihh_data = {}
        expert_data = {}
        with open(save_path + "sample_ihh_values.json", 'r') as j:
            ihh_data = json.load(j)
        experts = [(f.split('_')[0],  expert_path + f) for f in listdir(expert_path) if isfile(join(expert_path, f)) and
                   f.endswith("_thresholds.json")]
        sample_specific_data = ihh_data[sample_to_use]
        sample_specific_data['x'].reverse()
        sample_specific_data['y'].reverse()
        intense_min = sample_specific_data['x'][0] - 1
        for e in experts:
            with open(e[1]) as j:
                expert_results = json.load(j)
                if sample_to_use in expert_results:
                    expert_data[e[0]] = expert_results[sample_to_use]["high"]
        '''colour_range = sns.color_palette('tab10')[1:]
        colour_counter = 0
        for ex, thrsh in expert_data.items():
            plt.axvline(x=thrsh, label=ex, color=colour_range[colour_counter])
            colour_counter += 1'''
        expert_values = [val for exp, val in expert_data.items()]
        exp_min = int(min(expert_values))
        exp_max = int(max(expert_values))
        thresh_max = sample_specific_data['y'][np.where(np.array(sample_specific_data['x']) == exp_min)[0][0]]
        thresh_min = sample_specific_data['y'][np.where(np.array(sample_specific_data['x']) == exp_max)[0][0]]
        index_max = np.where(np.array(sample_specific_data['x']) == exp_min)[0][0]
        index_min = np.where(np.array(sample_specific_data['x']) == exp_max)[0][0]
        thresh_diffs = [max(sample_specific_data['y'])*0.05, thresh_max - thresh_min]
        highlight_distrib_max = np.array(sample_specific_data['y'])[index_max:index_min] + thresh_diffs[0]
        highlight_distrib_min = np.array(sample_specific_data['y'])[index_max:index_min] - thresh_diffs[0]
        plt.fill_between(np.arange(exp_min, exp_max), highlight_distrib_max, highlight_distrib_min, color='g', alpha=0.7)
        sns.lineplot(x=sample_specific_data['x'], y=sample_specific_data['y'])
        plt.show()
        slopes, slope_points = self._get_slope(sample_specific_data['x'], sample_specific_data['y'])
        mving_slopes = self._moving_average(slopes, window_size=8)
        '''colour_range = sns.color_palette('tab10')[1:]
        colour_counter = 0
        for ex, thrsh in expert_data.items():
            plt.axvline(x=thrsh, label=ex, color=colour_range[colour_counter])
            colour_counter += 1
        sns.lineplot(x=slope_points, y=slopes)
        plt.show()
        colour_range = sns.color_palette('tab10')[1:]
        colour_counter = 0
        for ex, thrsh in expert_data.items():
            plt.axvline(x=thrsh, label=ex, color=colour_range[colour_counter])
            colour_counter += 1
        sns.lineplot(x=slope_points, y=mving_slopes)
        plt.show()'''

        sample_image = io.imread(self.image_paths[sample_to_use])
        low_thresh_only = self._threshold_image(sample_image, intense_min, intense_min + 1)*sample_image
        lower_high_change = self._threshold_image(sample_image, intense_min, exp_min)*sample_image
        higher_high_change = self._threshold_image(sample_image, intense_min, exp_max)*sample_image
        flattened_rgb = np.amax(np.stack([low_thresh_only, lower_high_change, higher_high_change], axis=-1), axis=0)
        io.imshow(flattened_rgb)
        plt.show()
        max_slope = math.ceil(max(mving_slopes))
        logist = self._generate_sigmoid(max_slope / 2, k=6)
        logist_rescaled = np.array([logist[int(lgr)] for lgr in mving_slopes])
        print(len(logist), logist_rescaled.shape, max_slope)
        logist_weighted = self._apply_weights(logist_rescaled, mving_slopes)
        logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted, S=0.1, curve="convex",
                                    direction="decreasing")
        logist_knee = int(logist_knee_f.knee)
        print("Weighted Knee", logist_knee)
        '''sns.lineplot(x=sample_specific_data['x'], y=sample_specific_data['y'])
        plt.show()
        sns.lineplot(x=slope_points, y=mving_slopes)
        plt.show()
        sns.lineplot(x=slope_points[logist_knee:], y=mving_slopes[logist_knee:])
        plt.show()'''
        print(len(mving_slopes[logist_knee:]), len(mving_slopes), logist_knee)
        '''sns.lineplot(x=np.arange(len(logist)), y=logist)
        plt.show()'''
        '''sns.lineplot(x=slope_points, y=logist_rescaled)
        plt.show()'''
        norm_ihh = [y/max(sample_specific_data['y']) for y in sample_specific_data['y']]
        centr = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist, norm_ihh[logist_knee:-1], weight_option=2)
        print("Centroid", centr + logist_knee)
        centr_2 = self._logistic_thresholding(mving_slopes, sample_specific_data['y'], steepness=6, weighted_option=2)
        print("Centroid 2", centr_2 + logist_knee)
        rensu_im = self._threshold_image(sample_image, intense_min, expert_data['rensu'])*sample_image
        auto_im = self._threshold_image(sample_image, intense_min, intense_min + centr)*sample_image
        rgb_2 = np.amax(np.stack([rensu_im, auto_im, sample_image], axis=-1), axis=0)
        io.imshow(rgb_2)
        plt.show()

    def vox_struct_info(self, sample_used):
        expert_path = "C:\\RESEARCH\\Mitophagy_data\\gui params\\"
        expert_data = {}
        experts = [(f.split('_')[0], expert_path + f) for f in listdir(expert_path) if isfile(join(expert_path, f)) and
                   f.endswith("_thresholds.json")]
        for e in experts:
            with open(e[1]) as j:
                expert_results = json.load(j)
                if sample_used in expert_results:
                    expert_data[e[0]] = expert_results[sample_used]["high"]

        spatial_info_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\spatio_struct_info.json"
        with open(spatial_info_path, "r") as j:
            spatial_info = json.load(j)
        sample_specific_info = spatial_info[sample_used]
        low_thrsh = sample_specific_info["low_thr"]
        #print("Low thresh", low_thrsh)
        ihh_data = sample_specific_info["ihh"]
        ihh_data.reverse()
        norm_ihh = (np.array(ihh_data) / max(ihh_data))[:-1]
        intensities = np.arange(int(low_thrsh), int(low_thrsh) + int(len(ihh_data)))
        struct_info = sample_specific_info["Structures Lost"]
        structures_present = []
        current_total_mean = []
        current_structures = None
        prior_structures = None
        struct_size_change = []
        struct_size_past = None
        struct_size_gradient = []
        struct_size_gradient_prior = None
        struct_count = []
        added_struct_count = []
        average_struct_size_present = []
        voxels_added = []
        for g in range(len(struct_info) - 1, 0, -1):
            # print(current_structures, prior_structures, len(struct_info[g]))
            if current_structures is None and len(struct_info[g]) > 0:
                current_structures = np.array(struct_info[g])
                prior_structures = np.array(struct_info[g])
                struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                struct_size_change.append(struct_size_past)
                struct_size_gradient.append(struct_size_gradient_prior)
            elif len(struct_info[g]) == 0 and struct_size_past is not None:
                struct_size_change.append(struct_size_past)
                struct_size_gradient.append(struct_size_gradient_prior)
            elif current_structures is not None and prior_structures is not None and struct_size_past is not None and len(struct_info[g]) > 0:
                current_structures = np.array(struct_info[g])
                struct_size_past = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean()) + struct_size_past
                struct_size_change.append(struct_size_past)
                struct_size_gradient_prior = ((current_structures.mean() - prior_structures.mean()) / current_structures.mean())
                struct_size_gradient.append(struct_size_gradient_prior)
                prior_structures = np.array(struct_info[g])
            else:
                pass
            if current_structures is not None:
                structures_present += struct_info[g]
                struct_count.append(len(structures_present))
                added_struct_count.append(len(struct_info[g]))
                current_total_mean.append(np.array(structures_present).mean())
                struct_sze = np.array(struct_info[g]).mean() if len(struct_info[g]) > 0 else 0
                average_struct_size_present.append(struct_sze)
                voxels_added.append(sum(struct_info[g]))
        struct_size_change.reverse()
        struct_size_gradient.reverse()
        current_total_mean.reverse()
        added_struct_count.reverse()
        struct_count.reverse()
        average_struct_size_present.reverse()
        voxels_added.reverse()
        cum_change_mean = np.array(struct_size_change).mean()
        leftmost_intersect_intensity = []
        for t in range(1, len(struct_size_change)):  # need to interpolate to determine actual intersections. go from left to right
            if struct_size_change[t - 1] < cum_change_mean < struct_size_change[t]:
                if abs(struct_size_change[t - 1] - cum_change_mean) < abs(struct_size_change[t] - cum_change_mean):
                    leftmost_intersect_intensity.append(t - 1)
                else:
                    leftmost_intersect_intensity.append(t)
            else:
                if cum_change_mean == struct_size_change[t - 1]:
                    leftmost_intersect_intensity.append(t - 1)
                if cum_change_mean == struct_size_change[t]:
                    leftmost_intersect_intensity.append(t)
        leftmost_intersect_intensity = min(leftmost_intersect_intensity)
        #print(len(struct_size_change), len(struct_size_gradient))
        left_peak_change = np.where(
            struct_size_gradient[0:leftmost_intersect_intensity + 1] == np.amin(struct_size_gradient[0:leftmost_intersect_intensity + 1]))
        left_peak_change = int(min(left_peak_change)[0])
        #print("Intersection point", left_peak_change + int(low_thrsh))
        #print("Expert Threshold", expert_data['rensu'])
        '''fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        sns.lineplot(y=struct_size_change, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=ax1)
        ax1.set_title("Cumulative change in mean struct size descending")
        # ax1.axhline(y=cum_change_mean, c='r')
        # ax1.axvline(x=left_peak_change + int(low_thrsh), c='g')
        sns.lineplot(y=struct_size_gradient, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_gradient)), ax=ax2)
        ax2.set_title("Relative change in average struct size between intensities descending")
        # ax2.axhline(y=np.array(struct_size_gradient).mean(), c='r')
        ax2.axvline(x=left_peak_change + int(low_thrsh), c='g')
        sns.lineplot(y=current_total_mean, x=np.arange(int(low_thrsh), int(low_thrsh) + len(current_total_mean)), ax=ax3)
        ax3.set_title("Mean structure size of all structures currently present")

        fig2, (ax4, ax5, ax6) = plt.subplots(3, 1)
        sns.lineplot(y=average_struct_size_present, x=np.arange(int(low_thrsh), int(low_thrsh) + len(average_struct_size_present)), ax=ax4)
        ax4.set_title("Average Structure size of newly added structures")
        sns.lineplot(y=voxels_added, x=np.arange(int(low_thrsh), int(low_thrsh) + len(voxels_added)), ax=ax5)
        ax5.set_title("Newly added Voxels")
        sns.lineplot(y=struct_count, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_count)), ax=ax6)
        ax6.set_title("Structures present")
        plt.show()'''

        def integr_centroid(number_range):
            test_centr_1 = np.trapz(y=number_range * np.arange(int(low_thrsh), left_peak_change + int(low_thrsh))) / np.trapz(number_range)
            return test_centr_1

        def adjust_remaining_curve(distrib, separator):
            array_representation = np.array(distrib)
            if array_representation.min() < 0:
                array_representation += abs(array_representation.min())
            adjusted_distrib = np.ones_like(array_representation)
            # print("Array Rep Shape 1", adjusted_distrib.shape)
            adjusted_distrib[:separator] = array_representation[:separator]
            # print("Array Rep Shape 2", adjusted_distrib.shape)
            constant_val = distrib[separator]
            max_range = len(distrib)
            step_points = constant_val/(max_range - separator)
            # print(constant_val, step_points, separator, max_range)
            decreasing_range = [constant_val - dr*step_points for dr in range(max_range - separator)]
            # print(decreasing_range)
            adjusted_distrib[separator:] *= 0
            # print("Array Rep Shape 3", adjusted_distrib.shape)
            centroid = np.trapz(y=adjusted_distrib * np.arange(int(low_thrsh), int(low_thrsh) + len(adjusted_distrib))) / np.trapz(adjusted_distrib)
            return centroid

        mean_size_by_count = np.array(current_total_mean)
        other_centr = np.trapz(y=mean_size_by_count * np.arange(int(low_thrsh), int(low_thrsh) + len(mean_size_by_count))) / np.trapz(mean_size_by_count)
        intersect = left_peak_change + int(low_thrsh)
        other_centr_adjusted = adjust_remaining_curve(mean_size_by_count, left_peak_change)
        #print("Mean struct size and count centroid", other_centr, other_centr_adjusted)
        mean_struct_centroid = integr_centroid(current_total_mean[:left_peak_change])
        cum_change_centroid = integr_centroid(struct_size_change[:left_peak_change])
        #print(mean_struct_centroid, cum_change_centroid)
        mean_struct_centroid_2 = adjust_remaining_curve(current_total_mean, left_peak_change)
        cum_change_centroid_2 = adjust_remaining_curve(struct_size_change, left_peak_change)
        #print(mean_struct_centroid_2, cum_change_centroid_2)

        '''sample_image = io.imread(self.image_paths[sample_used])
        expert_thresh = self._threshold_image(sample_image, int(low_thrsh), int(expert_data['rensu']))*sample_image
        intersect_thresh = self._threshold_image(sample_image, int(low_thrsh), intersect)*sample_image
        centroid_thresh = self._threshold_image(sample_image, int(low_thrsh), other_centr_adjusted)*sample_image
        flattened_rgb = np.amax(np.stack([expert_thresh, intersect_thresh, centroid_thresh], axis=-1), axis=0)
        io.imshow(flattened_rgb)
        plt.show()'''

        slopes, slope_points = self._get_slope(intensities, ihh_data)
        mving_slopes = self._moving_average(slopes, window_size=8)
        max_slope = math.ceil(max(mving_slopes))
        logist = self._generate_sigmoid(max_slope / 2, k=6)
        logist_rescaled = np.array([logist[int(lgr)] for lgr in mving_slopes])
        #print(len(logist), logist_rescaled.shape, max_slope)
        logist_weighted = self._apply_weights(logist_rescaled, mving_slopes)
        logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted, S=0.1, curve="convex",
                                    direction="decreasing")
        logist_knee = int(logist_knee_f.knee)
        print("Normal Centroids:")
        high_thresh_1 = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist, norm_ihh[logist_knee:], weight_option=2) + int(low_thrsh)
        #print("High threshold", centr + int(low_thrsh))
        #print("Sizes for distribution weightings", len(norm_ihh), len(mean_size_by_count))
        new_ihh_weighting = (mean_size_by_count / mean_size_by_count.max()) * norm_ihh
        print("Adjusted Centroids:")
        high_thresh_2 = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist, new_ihh_weighting[logist_knee:], weight_option=2) + int(low_thrsh)
        #print("With weighting adjustments", centr + int(low_thrsh))
        # Investigate centroid manipulations further to try to find a middle ground

        return intersect, expert_data['rensu'], low_thrsh, other_centr, other_centr_adjusted, high_thresh_1, high_thresh_2

    def get_vox_info_per_sample(self, save_path):
        vox_data = {}
        for f in self.file_list:
            print("Sample", f[1])
            intersect, expert_thresh, low_thresh, struct_centroid, struct_centroid_cut, high_thresh_normal, high_thresh_adjusted = self.vox_struct_info(f[1])
            vox_data[f[1]] = {"Intersection": str(intersect), "Expert":str(expert_thresh), "Low":str(low_thresh),
                              "Struct centroids - norm and adjust": [str(struct_centroid), str(struct_centroid_cut)],
                              "High - without and with":[str(high_thresh_normal), str(high_thresh_adjusted)]}
        '''with open(save_path + "vox_struct_effect3.json", 'w') as j:
            json.dump(vox_data, j)'''

    def test_reverse_windowing_more(self, sample_used):
        expert_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\"
        expert_data = {}
        experts = [(f.split('_')[0], expert_path + f) for f in listdir(expert_path) if isfile(join(expert_path, f)) and
                   f.endswith("_thresholds.json")]
        for e in experts:
            with open(e[1]) as j:
                expert_results = json.load(j)
                if sample_used in expert_results:
                    expert_data[e[0]] = expert_results[sample_used]["high"]

        ihh_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\sample_ihh_values.json"
        with open(ihh_path, "r") as j:
            ihh_info = json.load(j)
        sample_specific_info = ihh_info[sample_used]
        intensities = sample_specific_info["x"]
        ihh_data = sample_specific_info["y"]
        intensities.reverse()
        ihh_data.reverse()
        low_thrsh = intensities[0]-1

        intensities = np.arange(int(low_thrsh), int(low_thrsh) + int(len(ihh_data)))

        slopes, slope_points = self._get_slope(intensities, ihh_data)
        mving_slopes = self._moving_average(slopes, window_size=8)
        '''fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        sns.lineplot(y=ihh_data, x=intensities, ax=ax1)
        sns.lineplot(y=slopes, x=slope_points, ax=ax2)
        sns.lineplot(y=mving_slopes, x=slope_points, ax=ax3)
        plt.show()'''
        max_slope = math.ceil(max(mving_slopes))
        logist = self._generate_sigmoid(max_slope / 2, k=6)
        logist_rescaled = np.array([logist[int(lgr)] for lgr in mving_slopes])
        logist_weighted = self._apply_weights(logist_rescaled, mving_slopes)
        logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted, S=0.1,
                                    curve="convex",
                                    direction="decreasing")
        logist_knee = int(logist_knee_f.knee)
        norm_ihh = (np.array(ihh_data)[logist_knee:] / max(ihh_data[logist_knee:]))
        print("Normal Centroids:")
        high_thresh_1 = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist,
                                                              norm_ihh[:-1], weight_option=2) + int(low_thrsh)

        print("Reverse Centroids:")
        high_thresh_2 = self._weighted_intensity_centroid_rev(mving_slopes[logist_knee:], logist,
                                                              norm_ihh[:-1], weight_option=2) + int(
            low_thrsh)
        print("Normal Centroid windows", high_thresh_1, "Reverse Windows", high_thresh_2)
        print("---------------------Adjusted denominator------------------------")
        print("Normal Centroids:")
        high_thresh_1 = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist,
                                                              norm_ihh[:-1], weight_option=2, voxel_biasing=True) + int(low_thrsh)

        print("Reverse Centroids:")
        high_thresh_2 = self._weighted_intensity_centroid_rev(mving_slopes[logist_knee:], logist,
                                                              norm_ihh[:-1], weight_option=2, voxel_biasing=True) + int(
            low_thrsh)
        print("Normal Centroid windows", high_thresh_1, "Reverse Windows", high_thresh_2)
        print("")

    def go_through_samples(self):
        for f in self.file_list:
            print("Sample", f[1])
            self.test_reverse_windowing_more(f[1])

    def show_low_thresholds(self):
        json_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\System Metrics\\lw_thrsh_metrics.json"
        for f in self.file_list:
            print("Sample", f[1])
            with open(json_path, "r") as j:
                low_values = json.load(j)
            sample_image = io.imread(f[0])
            sample_metrics = low_values[f[1] + " 0"]
            intensity_counts, intensity_range = histogram(sample_image, nbins=256)
            sns.lineplot(x=intensity_range, y=intensity_counts)
            plt.axvline(x=int(sample_metrics["Otsu"]), c='r')
            plt.axvline(x=int(sample_metrics["Normal"]), c='g')
            plt.axvline(x=int(sample_metrics["Log"]), c='k')
            plt.show()

    def gradient_represents(self):
        json_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\sample_ihh_values.json"
        with open(json_path, "r") as j:
            ihh_information = json.load(j)
        for f in self.file_list:
            print("Sample", f[1])
            sample_information = ihh_information[f[1]]
            sample_information['x'].reverse()
            sample_information['y'].reverse()
            slopes, slope_points = self._get_slope(sample_information['x'], sample_information['y'])
            mving_slopes = self._moving_average(slopes, window_size=8)
            '''if f[1] == "CCCP_2C=0T=0.tif":
                print("Original Grads")
                print(slopes)
                print(slope_points)
                sns.lineplot(x=np.linspace(start=0, stop=len(slopes), num=len(slopes)), y=slopes)
                plt.show()
                average_test = self.moving_average_test(slopes, 8)
                print("Testing rolling average")
                print(average_test)
                fig1, (a1, a2) = plt.subplots(2)
                plt.tight_layout()
                sns.lineplot(x=slope_points, y=mving_slopes, ax=a1)
                sns.lineplot(x=slope_points, y=average_test, ax=a2)
                plt.show()'''
            inverted_values, some_dict = self._invert_rescaler(mving_slopes)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            plt.tight_layout()
            sns.lineplot(x=sample_information['x'], y=sample_information['y'], ax=ax1)
            ax1.set_title("Original IHH")
            sns.lineplot(x=slope_points, y=slopes, ax=ax2)
            ax2.set_title("Gradient Distribution")
            sns.lineplot(x=slope_points, y=mving_slopes, ax=ax3)
            ax3.set_title("Smoothed Gradient Distribution")
            sns.lineplot(x=slope_points, y=inverted_values, ax=ax4)
            ax4.set_title("Inverted Gradient Distribution")
            plt.show()
            sns.lineplot(x=sample_information['x'], y=sample_information['y'])
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Non-zero Voxel Count")
            plt.show()
            '''sns.lineplot(x=slope_points, y=np.array(slopes)/max(slopes))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Gradient")
            plt.show()'''
            sns.lineplot(x=slope_points, y=np.array(mving_slopes)/max(mving_slopes))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Gradient")
            plt.show()
            '''sns.lineplot(x=slope_points, y=np.array(inverted_values)/max(inverted_values))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverse Gradient")
            plt.show()'''
            sns.lineplot(x=slope_points, y=np.array(inverted_values)*np.array(sample_information['y'])[:-1])
            plt.show()
            integral_test = np.trapz(y=np.array(inverted_values) * np.array(slope_points))/np.trapz(np.array(inverted_values))
            bad_centre = np.trapz(y=np.array(inverted_values) * np.array(slope_points))/len(inverted_values)
            ihh_array = np.array(sample_information['y'])[:-1]
            normed_ihh = ihh_array/ihh_array.max()
            invert_centr = self._inverted_thresholding(slopes, ihh_array, 2) + slope_points[0]
            integral_test_2 = np.trapz(y=normed_ihh*np.array(inverted_values) * np.array(slope_points)) / np.trapz(
                np.array(inverted_values))
            print("Centroid by integration", integral_test+slope_points[0], integral_test_2+slope_points[0])
            print("Bad Centroid", bad_centre)
            print("Window Centroid", invert_centr)
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.axvline(x=integral_test+slope_points[0], c='k')
            plt.axvline(x=bad_centre + slope_points[0], c='r')
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverse Gradient")
            plt.show()
            sns.lineplot(x=slope_points, y=np.array(inverted_values)*np.array(sample_information['y'])[:-1])
            plt.axvline(x=integral_test_2 + slope_points[0], c='k')
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverse Gradient")
            # plt.axvline(x=invert_centr, c='r')
            plt.show()

    def compare_expert_with_failing(self):
        json_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\sample_ihh_values.json"
        expert_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\rensu_thresholds.json"
        mip_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Expert Comparison\\MIP\\"
        tif_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Expert Comparison\\TIFF\\"
        with open(json_path, "r") as j:
            ihh_information = json.load(j)
        with open(expert_path, "r") as j:
            expert_info = json.load(j)
        for f in self.file_list:
            print("Sample", f[1])
            expert_threshes = expert_info[f[1]]
            voxel_counts = ihh_information[f[1]]["y"]
            intensity_range = ihh_information[f[1]]["x"]
            intensity_range.reverse()
            voxel_counts.reverse()
            low_thrsh = intensity_range[0] - 1

            intensities = np.arange(int(low_thrsh), int(low_thrsh) + int(len(voxel_counts)))

            slopes, slope_points = self._get_slope(intensities, voxel_counts)
            mving_slopes = self._moving_average(slopes, window_size=8)
            max_slope = math.ceil(max(mving_slopes))
            logist = self._generate_sigmoid(max_slope / 2, k=6)
            logist_rescaled = np.array([logist[int(lgr)] for lgr in mving_slopes])
            logist_weighted = self._apply_weights(logist_rescaled, mving_slopes)
            logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted,
                                        S=0.1,
                                        curve="convex",
                                        direction="decreasing")
            logist_knee = int(logist_knee_f.knee)
            norm_ihh = (np.array(voxel_counts)[logist_knee:] / max(voxel_counts[logist_knee:]))
            print("Normal Centroids:")
            high_thresh_1 = self._weighted_intensity_centroid_eff(mving_slopes[logist_knee:], logist,
                                                                  norm_ihh[:-1], weight_option=2) + int(low_thrsh)
            sample_image = io.imread(f[0])
            expert_im = (self._threshold_image(sample_image, expert_threshes["low"], expert_threshes["high"])*sample_image).astype('uint8')
            auto_im = (self._threshold_image(sample_image, int(low_thrsh), high_thresh_1)*sample_image).astype('uint8')
            zeros_im = np.zeros_like(sample_image)
            rgb_im = np.stack([expert_im, auto_im, zeros_im], axis=-1)
            mip_im = np.amax(rgb_im, axis=0)
            io.imsave(mip_path + f[1].split('.')[0] + "_MIP.png", mip_im)
            io.imsave(tif_path + f[1], rgb_im)

    def generate_two_ihh_and_struct_metrics(self):
        expert_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\rensu_thresholds.json"
        low_thresh_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\System Metrics\\lw_thrsh_metrics.json"
        save_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\System Metrics\\"
        data_dictionary = {}
        with open(expert_path, "r") as j:
            expert_info = json.load(j)
        with open(low_thresh_path, "r") as j:
            low_thresh_details = json.load(j)
        for f in self.file_list:
            print("Sample", f[1])
            data_dictionary[f[1]] = {}
            expert_low = int(expert_info[f[1]]["low"])
            auto_details = low_thresh_details[f[1] + " 0"]
            auto_low = int(auto_details["Chosen"])
            sample_image = io.imread(f[0])
            expert_details, auto_details = self._efficient_hysteresis_iterative_pair(sample_image, expert_low, auto_low)
            data_dictionary[f[1]]["Expert"] = {"Intensities":expert_details[0], "Thresholds":expert_details[1],
                                               "Str_Counts":expert_details[2], "Str_Sizes":expert_details[3]}
            data_dictionary[f[1]]["Auto"] = {"Intensities": auto_details[0], "Thresholds": auto_details[1],
                                               "Str_Counts": auto_details[2], "Str_Sizes": auto_details[3]}
            with open(save_path + "many_metrics.json", 'w') as j:
                json.dump(data_dictionary, j)

    def moving_average_test(self, dist, window_size):
        offset = int(window_size / 2)
        odd_window = int(window_size % 2 != 0) #This odd window offset will include an extra right side value
        padding = [0] * offset
        windows = np.zeros(shape=tuple([len(dist), window_size]))
        dist = padding + dist + padding
        dist = np.array(dist)
        for n in range(offset, len(dist) - offset):
            beginning = n-offset
            ending = n+offset+odd_window
            windows[n-offset] = dist[beginning:ending]
        print(windows)
        window_averages = np.mean(windows, axis=1)
        return window_averages

    def highlight_flatter_regions(self):
        ihh_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\sample_ihh_values.json"
        #print(plt.get_cmap("coolwarm"))
        #print(plt.get_cmap("viridis").colors)
        viridis = plt.get_cmap("viridis").colors
        viridis.reverse()
        '''cw = plt.cm.coolwarm
        colour_segments = cw._segmentdata
        sns_colors = sns.color_palette("viridis")
        print(len(sns_colors))
        for rgb_, values in colour_segments.items():
            print("Colour", rgb_, "Variations", values)'''
        # viridis.reverse()
        with open(ihh_path, "r") as j:
            ihh_data = json.load(j)
        for sample, sample_ihh in ihh_data.items():
            print("Sample:", sample)
            voxel_counts = sample_ihh["y"]
            intensity_range = sample_ihh["x"]
            slopes, slope_points = self._get_slope(intensity_range, voxel_counts)
            mving_slopes = self._moving_average(slopes, window_size=8)
            inverted_values, some_dict = self._invert_rescaler(mving_slopes)
            norm_smoothed = (np.array(mving_slopes)/max(mving_slopes)).tolist()
            # Now to average these inverted gradients across the range. The first item will be between 0 and nothing
            associated_change = [None]*len(voxel_counts)
            print(len(inverted_values), len(voxel_counts))
            for g in range(len(voxel_counts)):
                if g == len(voxel_counts) - 1:
                    intermediate_change = norm_smoothed[g - 1]
                elif g > 0:
                    intermediate_change = (norm_smoothed[g] + norm_smoothed[g - 1]) / 2
                else:
                    intermediate_change = norm_smoothed[g]
                associated_change[g] = intermediate_change
            '''fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            fig.suptitle(sample)
            sns.lineplot(x=intensity_range, y=voxel_counts, ax=ax1)
            ax1.set_title("Voxel Counts")
            sns.lineplot(x=intensity_range, y=associated_change, ax=ax2)
            ax2.set_title("Change associated to thresholds")
            sns.lineplot(x=slope_points, y=inverted_values, ax=ax3)
            ax3.set_title("Inverted Gradient Distribution")
            plt.show()'''
            ratio = max(voxel_counts) * 0.01
            print("Highlight width", ratio)
            associated_change = np.array(associated_change)
            '''associated_change *= 0.8
            associated_change += 0.2'''
            sns.lineplot(x=intensity_range, y=voxel_counts, c='k')
            for t in range(len(intensity_range)-1):
                colour_index = associated_change[t]*255
                colour_ = viridis[int(colour_index)]
                alpha = 0.95 + 0.05 * (colour_index - int(colour_index))
                #alpha = 1
                plt.fill_between(np.array(intensity_range)[t:t+2], np.array(voxel_counts)[t:t+2]+ratio,
                                 np.array(voxel_counts)[t:t+2]-ratio, color=colour_, alpha=alpha)
            norm = plt.cm.colors.Normalize(0, 1)
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis")), label="Localized Gradient")
            plt.ylabel("Non-zero Voxel Count")
            plt.xlabel("High Thresholds")
            plt.show()

    def get_border_mip(self):
        image_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"
        sample_name = "CCCP_2C=0T=0.tif"
        z = 4
        sample_image = io.imread(image_path + sample_name)
        mip_original = np.amax(sample_image, axis=0)
        original_with_zero = (sample_image > 12)
        low_thresh = 23
        high_thresh = 45
        threshold_sample = self._threshold_image(sample_image, low_thresh, high_thresh)[z]
        without_low = (sample_image > low_thresh)[z]
        threshold_exlusion = np.logical_xor(threshold_sample, without_low).astype('uint8')
        '''fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax2.imshow(threshold_sample.astype('uint8')*sample_image[2])
        ax1.imshow(without_low.astype('uint8')*sample_image[2])
        overlay = np.stack([threshold_sample.astype('uint8')*sample_image[2], threshold_exlusion*sample_image[2], np.zeros_like(threshold_exlusion)], axis=-1)
        ax3.imshow(overlay*255)
        plt.show()'''
        low_thresholding_special = self._threshold_image(sample_image, 13, high_thresh)[z]
        low_thresholding_special = np.logical_xor(np.logical_or(low_thresholding_special, without_low), without_low)
        #low_thresholding_special = np.logical_xor(low_thresholding_special, threshold_sample).astype('uint8')
        io.imshow(np.stack([threshold_sample.astype('uint8'), threshold_exlusion, low_thresholding_special], axis=-1)*255)
        plt.show()
        grayscale = threshold_sample.astype('uint8')*255 + threshold_exlusion*155 + low_thresholding_special*25
        io.imshow(grayscale, cmap='gray')
        '''grey_range = plt.cm.get_cmap("binary")
        grey_range.reversed()'''
        # io.imshow(np.stack([grayscale, grayscale, grayscale], axis=-1))
        plt.show()

    def generate_ihh_colour_rep(self):
        for f in self.file_list:
            print("Sample", f[1])
            image = self._grayscale(io.imread(f[0]))
            lw_thrsh = self._low_select(img=image)[0]
            print("Low Threshold", lw_thrsh)
            print("Intensity Range", list(range(lw_thrsh+1, image.max()+1)))
            mask_low = image > lw_thrsh
            labels_low, num_labels = ndi.label(mask_low)
            valid_structures = np.stack([labels_low, image*(mask_low.astype('int'))], axis=-1) # The two labels have been stacked
            valid_structures = np.reshape(valid_structures, (-1, valid_structures.shape[-1])) # The dual label image has been flattened save for the label pairs
            #valid_structures = valid_structures[np.nonzero(valid_structures[:, 0])] # The zero label structs have been removed
            sort_indices = np.argsort(valid_structures[:, 0])
            valid_structures = valid_structures[sort_indices]
            label_set, start_index, label_count = np.unique(valid_structures[:, 0], return_index=True, return_counts=True)
            end_index = start_index + label_count
            max_labels = np.zeros(tuple([len(label_set), 2]))
            canvas_image = np.zeros_like(labels_low)
            for t in range(len(label_set)):
                max_labels[t, 0] = label_set[t]
                max_labels[t, 1] = valid_structures[slice(start_index[t], end_index[t]), 1].max()
                # canvas_image += (labels_low == label_set[t]).astype('int') * valid_structures[slice(start_index[t], end_index[t]), 1].max()
            value_mapping = max_labels[:, 1]
            canvas_image = value_mapping[labels_low]
            def build_ihh():
                intensities, voxel_count = np.unique(canvas_image, return_counts=True)
                intensities, voxel_count = intensities[1:], voxel_count[1:]
                intense_range = int(intensities[-1]-intensities[0]+2)
                all_intense_voxels = np.zeros(tuple([intense_range]))
                indexing_arr = (intensities - intensities[0]).astype(int)
                all_intense_voxels[indexing_arr] = voxel_count
                all_intense_voxels = all_intense_voxels[1:]
                voxel_accum = np.flip(np.cumsum(np.flip(all_intense_voxels)))
                intensities = np.linspace(intensities[0], intensities[-1], int(intensities[-1] - intensities[0] + 1))
                return intensities, voxel_accum
            x_val, y_val = build_ihh()
            sns.lineplot(x=x_val, y=y_val)
            plt.show()

            # Use slice(start_index, end_index) but will need to be looped
            '''invalid_pairs = np.logical_not(np.any(reordered == 0, axis=-1))
            pairings, pairing_sizes = np.unique(reordered[invalid_pairs], return_counts=True, axis=0)'''


    def new_figures(self, sample_name=None):
        '''
        - Inverted grad with black dashed line for centroid (bad centroid) *
        - Same as above but red dashed-dotted line for better centroid *
        - Inverted grad with no bias, window bias (option 0), IHH bias, and both bias. Vertical dashed line for centroid
        - Inverted grad with black dashed line for no window bias and red dash dotted line for window bias.
        - Inverted grad with different coloured boxes with height differences for each window (a subset of windows) and
        a vertical line for each centroid
        - a distribution of window centroids (y-axis) by window widths (x-axis)
        - black and red vertical lines for centroid without and with window bias. Show for centroid distribution vs
         window width for each sample (2 samples)
        - centroid for inverted grad, centroid for ihh, overlay of these two with a combined centroid
        - sample with IHH overlaid and bias applied with annotated centroid. Have normal ihh bias and softened bias
        (2 samples)
        - mip of sample at specific high thresholds to annotate onto the point to show what the difference is. Show
        intensity representation and then the overlaid must be binary of each color e.g red and blue (perhaps some kind
        of hatching could be used for differentiation.
        - an overlay of layers representing a range of 4 high threshold steps (above and below) to show the impact of
        the smoothing taking further context into account.
        :return:
        '''

        json_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\IHH_Series\\sample_ihh_values.json"
        with open(json_path, "r") as j:
            ihh_information = json.load(j)

        def graph_sample(sample_file):
            print("Sample", sample_file[1])
            sample_information = ihh_information[sample_file[1]]
            sample_information['x'].reverse()
            sample_information['y'].reverse()
            slopes, slope_points = self._get_slope(sample_information['x'], sample_information['y'])
            low_thresh = slope_points[0] - 1
            print("Low Thresh", low_thresh)
            mving_slopes = self._moving_average(slopes, window_size=8)
            inverted_values, some_dict = self._invert_rescaler(mving_slopes)
            bad_centre = np.trapz(y=np.array(inverted_values) * np.array(slope_points)) / len(inverted_values)
            correct_centroid = np.trapz(y=np.array(inverted_values) * np.array(slope_points)) / np.trapz(
                np.array(inverted_values))
            manual_test_num = []
            manual_test_den = []
            for r in range(0, len(inverted_values)):
                manual_test_num.append(inverted_values[r]*slope_points[r])
                manual_test_den.append(inverted_values[r])
            ihh_array = np.array(sample_information['y'])[:-1]
            normed_ihh = ihh_array / ihh_array.max()
            only_ihh = np.trapz(y=normed_ihh * np.array(inverted_values) * np.array(slope_points)) / np.trapz(
                np.array(inverted_values))
            no_ihh = np.ones_like(ihh_array)
            window_only = self._inverted_thresholding(mving_slopes, no_ihh, 1, 0) + low_thresh
            both_bias = self._inverted_thresholding(mving_slopes, ihh_array, 1, 0) + low_thresh
            '''sns.lineplot(x=sample_information['x'], y=sample_information['y'])
            plt.show()
            fig, (ax1, ax2) = plt.subplots(2, 1)
            sns.lineplot(x=slope_points, y=slopes, ax=ax1)
            sns.lineplot(x=slope_points, y=mving_slopes, ax=ax2)
            plt.show()'''
            print("Inverted value with window only", window_only)
            # Inverted with just bad centroid
            print("Bad centroid Inverted")
            # adjust axis label font size using plt.xlabel("blahblah", fontsize=18)
            '''sns.lineplot(x=slope_points, y=inverted_values)
            plt.axvline(x=bad_centre, c='k', dashes=(5, 2, 5, 2))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.show()
            # correct centroid
            print("Correct Centroid")
            sns.lineplot(x=slope_points, y=inverted_values)
            #plt.axvline(x=bad_centre, c='k', dashes=(5, 2, 5, 2))
            plt.axvline(x=correct_centroid, c='r', dashes=(6, 2, 2, 2))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.show()
            # both centroids
            print("Good and Bad Centroids")
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.axvline(x=bad_centre, c='k', dashes=(5, 2, 5, 2))
            plt.axvline(x=correct_centroid, c='r', dashes=(6, 2, 2, 2))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.show()
            # window centroid
            print("Window Bias only")
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.axvline(x=window_only, c='k', dashes=(5, 2, 5, 2))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.show()
            # ihh only
            print("IHH Bias Only")
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.axvline(x=only_ihh, c='k', dashes=(5, 2, 5, 2))
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.show()
            # both bias
            print("IHH and Window Applied")
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.axvline(x=both_bias, c='k', dashes=(5, 2, 5, 2))
            plt.show()
            # with and without window
            print("With and without window overlaid")
            sns.lineplot(x=slope_points, y=inverted_values)
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Inverted Non-zero Voxel Count Change")
            plt.axvline(x=correct_centroid, c='k', dashes=(5, 2, 5, 2))
            plt.axvline(x=window_only, c='r', dashes=(6, 2, 2, 2))
            plt.show()
            # window data
            consolidated_centroid, window_centroids = self._inverted_thresholding_with_windows(mving_slopes, no_ihh, 1, 0)
            consolidated_centroid = consolidated_centroid + low_thresh
            print("The window centroids over window sizes")
            window_sizes = np.array(list(window_centroids.keys()))
            window_centr = np.array(list(window_centroids.values())) + low_thresh
            sns.lineplot(x=window_sizes, y=window_centr)
            plt.xlabel("Window Width")
            plt.ylabel("Centre of Mass Value")
            plt.show()
            # ihh centroid (not bias)
            ihh_centre = np.trapz(y=normed_ihh * np.array(slope_points)) / np.trapz(
                np.array(normed_ihh))
            print("Norm IHH Centroid")
            sns.lineplot(x=slope_points, y=normed_ihh)
            plt.xlabel("High Threshold Intensities")
            plt.ylabel("Normalized Non-zero Voxel Count")
            plt.axvline(x=ihh_centre, c='k', dashes=(5, 2, 5, 2))
            plt.show()
            # both bias with overlaid distribution
            print("Overlaid with colours")
            colours = sns.color_palette()
            print(type(colours))
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            sns.lineplot(x=slope_points, y=inverted_values, ax=ax1, color=colours[0])
            sns.lineplot(x=slope_points, y=normed_ihh, ax=ax2, color=colours[1])
            plt.axvline(x=both_bias, c='k', dashes=(5, 2, 5, 2))
            ax1.set_xlabel("High Threshold Intensities")
            ax1.set_ylabel("Inverted Non-zero Voxel Count Change", color=colours[0])
            ax2.set_ylabel("Normalized Non-zero Voxel Count", color=colours[1])
            plt.show()
            # normal and softened ihh bias
            print("softened ihh")
            softened_ihh_array = np.sqrt(normed_ihh)
            softened_ihh_array = softened_ihh_array / softened_ihh_array.max()
            softened_ihh = np.trapz(y=softened_ihh_array * np.array(inverted_values) * np.array(slope_points)) / np.trapz(
                np.array(inverted_values))
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            sns.lineplot(x=slope_points, y=inverted_values, color=colours[0], ax=ax1)
            sns.lineplot(x=slope_points, y=softened_ihh_array, color=colours[1], ax=ax2)
            ax1.set_xlabel("High Threshold Intensities")
            ax1.set_ylabel("Inverted Non-zero Voxel Count Change", color=colours[0])
            ax2.set_ylabel("Normalized Non-zero Voxel Count", color=colours[1])
            plt.axvline(x=correct_centroid, c='k', dashes=(5, 2, 5, 2))
            plt.axvline(x=softened_ihh, c='r', dashes=(6, 2, 2, 2))
            plt.show()
            print("Normal IHH")
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            sns.lineplot(x=slope_points, y=inverted_values, color=colours[0], ax=ax1)
            sns.lineplot(x=slope_points, y=normed_ihh,  color=colours[1], ax=ax2)
            plt.axvline(x=correct_centroid, c='k', dashes=(5, 2, 5, 2))
            plt.axvline(x=only_ihh, c='r', dashes=(6, 2, 2, 2))
            ax1.set_xlabel("High Threshold Intensities")
            ax1.set_ylabel("Inverted Non-zero Voxel Count Change", color=colours[0])
            ax2.set_ylabel("Normalized Non-zero Voxel Count", color=colours[1])
            plt.show()'''

            def get_im_at_intensities(thresh_intensities):
                raw_image = io.imread(sample_file[0])
                for ht in thresh_intensities:
                    binary_rep = self._threshold_image(raw_image, low_thresh, ht).astype('uint8')
                    print("Image at", ht)
                    io.imshow(np.amax(binary_rep*raw_image, axis=0))
                    plt.show()

            get_im_at_intensities([56, 84, 140, 180, 212, 250])

            def get_sample_ihh_points(high_thresholds):
                raw_image = io.imread(sample_file[0])
                image_canvas = np.zeros_like(raw_image)
                for ht in high_thresholds:
                    binary_rep = self._threshold_image(raw_image, low_thresh, ht).astype('uint8')
                    image_canvas += binary_rep
                flattened_rep = np.amax(image_canvas, axis=0)
                io.imshow(flattened_rep)
                plt.show()
            get_sample_ihh_points([33, 51, 107])

            def get_grad_discrete(high_thresh_range):
                raw_image = io.imread(sample_file[0])
                image_canvas = []
                for ht in range(high_thresh_range[0], high_thresh_range[1]):
                    thresh1 = self._threshold_image(raw_image, low_thresh, ht-1).astype('uint8')
                    thresh2 = self._threshold_image(raw_image, low_thresh, ht).astype('uint8')
                    overlay = np.amax(thresh1 + thresh2, axis=0)
                    image_canvas.append(overlay)
                for t in range(len(image_canvas)):
                    print("Threshold", high_thresh_range[0]-1+t, " to", high_thresh_range[0]+t)
                    io.imshow(image_canvas[t])
                    plt.show()
                '''panels = len(image_canvas)
                rows = math.ceil(panels/2)
                fig, axs = plt.subplots(rows, 2)
                fig.tight_layout()
                for t in range(panels):
                    row_index = int(t/2)
                    col_index = t - row_index*2
                    axs[row_index, col_index].imshow(image_canvas[t])
                plt.show()'''

            get_grad_discrete([186, 192])



        if sample_name is None:
            for f in self.file_list:
                graph_sample(sample_file=f)
        else:
            graph_sample(sample_file=[self.image_paths[sample_name], sample_name])

    def presentation_image(self):
        image_path = self.image_paths["CCCP_2C=0T=0.tif"]
        image = io.imread(image_path)
        orig_mip = np.amax(image, axis=0)
        '''io.imshow(orig_mip)
        plt.show()'''
        low_only_mask = (orig_mip <= 25).astype(int)
        low_only_mip = low_only_mask
        '''io.imshow(low_only_mip)
        plt.show()'''
        low_removed_mip = (orig_mip > 25).astype(int)
        '''io.imshow(low_removed_mip)
        plt.show()'''
        high_only_mip = (orig_mip > 55).astype(int)
        '''io.imshow(high_only_mip)
        plt.show()'''
        thresholded = self._threshold_image(image, 25, 55)
        thresh_mip = np.amax(thresholded, axis=0)
        '''io.imshow(thresh_mip)
        plt.show()'''
        save_location = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Presentation Images\\thresh_examples\\"

        '''io.imsave(save_location + "Orig_mip.png", orig_mip)

        io.imsave(save_location + "below_mask.png", low_only_mip)
        io.imsave(save_location + "below_mip.png", low_only_mip*orig_mip)

        io.imsave(save_location + "above_low_mask.png", low_removed_mip)
        io.imsave(save_location + "above_low_mip.png", low_removed_mip*orig_mip)

        io.imsave(save_location + "high_mask.png", high_only_mip)
        io.imsave(save_location + "high_mip.png", high_only_mip*orig_mip)

        io.imsave(save_location + "thresh_mask.png", thresh_mip)
        io.imsave(save_location + "thresh_mip.png", thresh_mip*orig_mip)'''

        intens_counts, intens_bins = histogram(image, nbins=256)
        sns.lineplot(x=intens_bins, y=intens_counts)
        plt.ylabel("Count of voxels per intensity")
        plt.xlabel("Intensity Values")
        plt.title("Histogram")
        plt.show()
        intens_counts, intens_bins = histogram(image, nbins=256)
        sns.lineplot(x=np.log(intens_bins+1), y=np.log(intens_counts+1))
        plt.ylabel("Count of voxels per intensity")
        plt.xlabel("Intensity Values")
        plt.title("Log of Histogram")
        plt.show()


if __name__ == "__main__":
    input_path = ["C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"]
    system_analyst = thresholding_metrics(input_path)
    # system_analyst.presentation_image()
    # system_analyst.new_figures("CCCP_1C=1T=0.tif")
    system_analyst.generate_ihh_colour_rep()
    # In the "many_metrics" json file the intensities and the voxel counts (Thresholds) are reverse
    # system_analyst.get_border_mip()
    # system_analyst.highlight_flatter_regions()
    # system_analyst.generate_two_ihh_and_struct_metrics()
    # system_analyst.compare_expert_with_failing()
    # system_analyst.gradient_represents()
    # system_analyst.show_low_thresholds()
    # system_analyst.go_through_samples()
    # system_analyst.get_vox_info_per_sample("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\")
    # system_analyst.vox_struct_info("CCCP_1C=1T=0.tif")
    # system_analyst.place_expert_lines("CCCP_1C=0T=0.tif")
    # system_analyst.place_expert_lines("CCCP_1C=1T=0.tif")
    # system_analyst.generate_ihh_figures()
    # system_analyst.get_sample_ihh()
    # system_analyst.recalc_thresh_with_vox_weight_new()
    # system_analyst.visualise_spatial_info()
    # system_analyst.get_spatial_metrics_more()
    # system_analyst.visualise_struct_counts()
    # system_analyst.extract_information_samples()
    # system_analyst.generate_IHH_plots("CCCP_1C=1T=0.tif")
    # system_analyst.go_through_image_ihh()
    # system_analyst.generate_ihh_figure(input_path[0] + "CCCP_1C=1T=0.tif", 26)
    # system_analyst.generate_threshold_graphs()
    # system_analyst.low_threshold_similarity("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\Low Threshold Metrics\\lw_thrsh_metrics.json")
    # system_analyst.save_histogram("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\CCCP_2C=1T=0.tif", 7)
    # system_analyst.large_excluded_test()
    # system_analyst.distribution_from_target()
    # system_analyst.high_and_low_testing()
    # system_analyst.structure_hunting()
    # system_analyst.stack_hist_plot()
    # system_analyst.compare_thresholds_between()
    # system_analyst._structure_overlap_test()
    # print(system_analyst.exp_threshes)
    # system_analyst.analyze_low_thresholds(save_path="C:\\Users\\richy\Desktop\\SystemAnalysis_files\\System Metrics\\")

