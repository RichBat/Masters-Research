import json
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
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
            plt.plot(intens, thresh2)
            plt.show()
            print("Threshold results")
            # print(thresh1)
            print(thresh2)
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
            struct_size_change.reverse()
            struct_size_gradient.reverse()
            current_total_mean.reverse()
            added_struct_count.reverse()
            struct_count.reverse()
            average_struct_size_present.reverse()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            sns.lineplot(y=struct_size_change, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_change)), ax=ax1)
            ax1.set_title("Cumulative change in mean struct size descending")
            ax1.axhline(y=np.array(struct_size_change).mean(), c='r')
            sns.lineplot(y=struct_size_gradient, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_size_gradient)), ax=ax2)
            ax2.set_title("Relative change in average struct size between intensities descending")
            ax2.axhline(y=0, c='k')
            ax2.axhline(y=np.array(struct_size_gradient).mean(), c='r')
            sns.lineplot(y=current_total_mean, x=np.arange(int(low_thrsh), int(low_thrsh) + len(current_total_mean)), ax=ax3)
            ax3.set_title("Mean structure size of all structures currently present")

            fig2, (ax4, ax5, ax6) = plt.subplots(3, 1)
            sns.lineplot(y=average_struct_size_present, x=np.arange(int(low_thrsh), int(low_thrsh) + len(average_struct_size_present)), ax=ax4)
            ax4.set_title("Average Structure size of newly added structures")
            sns.lineplot(y=struct_count, x=np.arange(int(low_thrsh), int(low_thrsh) + len(struct_count)), ax=ax5)
            ax5.set_title("Total number of independent structures")
            sns.lineplot(y=added_struct_count, x=np.arange(int(low_thrsh), int(low_thrsh) + len(added_struct_count)), ax=ax6)
            ax6.set_title("Independent structures added at this intensity")
            plt.show()


if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    system_analyst = thresholding_metrics(input_path)
    system_analyst.visualise_spatial_info()
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

