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
        starting_threshes = (20, 140)
        ending_threshes = (100, 200)

        def test_structure_thresh(low_thresh, high_thresh):
            return self._threshold_image(reduced_canvas, low_thresh, high_thresh)

        steps = (40, 30)

        starting_image = test_structure_thresh(starting_threshes[0], starting_threshes[1])
        starting_labels, _t = ndi.label(starting_image)
        low_res = np.arange(starting_threshes[0], ending_threshes[0]+1, steps[0])
        high_res = np.arange(starting_threshes[1], ending_threshes[1]+1, steps[1])
        result_array = np.zeros(tuple([len(low_res), len(high_res)]))
        thresholding_combos = np.stack([(np.ones_like(result_array).T * low_res).T, np.ones_like(result_array) * high_res], axis=-1)
        iteration_order = np.zeros(tuple([len(low_res), len(high_res)]))
        indice_array = np.argwhere(iteration_order == 0).sum(axis=1)/2
        iteration_order[np.nonzero(iteration_order + 1)] = indice_array
        k = max(len(low_res), len(high_res))
        iter_step_size = 0.5
        iteration_ranges = np.arange(iter_step_size, np.max(iteration_order) + iter_step_size, iter_step_size)
        result_shape_array = np.zeros(tuple([len(low_res), len(high_res), 2]))  # this is just for testing. Will store the shape of the overlap ratios
        store_neighbours = []  # purely for testing as this is less efficient than arrays and in practice neighbourhood inheritance will happen in loop
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
            # ***********************
            for tp in np.argwhere(viable_elements):
                threshold_params = thresholding_combos[tp[0], tp[1]]  # should assign a 1d array of two elements where ele 0 is low and ele 1 is high
                thresholded_image = test_structure_thresh(threshold_params[0], threshold_params[1])
                if np.equal(starting_image, thresholded_image).all():
                    print("These should not be identical")
                over_ratio, vol_ratio, structure_pairs, excluded = self._structure_overlap(starting_image, thresholded_image)
                starting_image = thresholded_image  # need to make sure that this is not linked (shallow copy ?!!) so it doesn't mutate
                if over_ratio.shape != vol_ratio.shape:
                    print("Ratio shape unmatched")
                    print(over_ratio.shape, vol_ratio.shape)
                print("Excluded structures", excluded)
                result_shape_array[tp[0], tp[1], 0] = over_ratio.shape[0]
                result_shape_array[tp[0], tp[1], 1] = over_ratio.shape[1]

        print(result_shape_array)
        print(iteration_order)
        print(result_shape_array.shape)
        print("Going through neighbours")

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
        io.imshow(image_21+image_20)
        plt.show()
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

        '''over_02, store02 = adjust_values_for_pairs(over_0001, over_0102)
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
        print(over_12a.sum(axis=1), over_12b.sum(axis=1))'''
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
        '''print("Overlap regions:", overlap_count)
        io.imshow(overlap_regions)
        plt.show()'''
        if not labels_provided:
            binary1 = image1 > 0
            binary2 = image2 > 0
            structure_seg1, structure_count1 = ndi.label(binary1)  # this labeled array should be an argument
            structure_seg2, structure_count2 = ndi.label(binary2)  # same for this labeled array
        else:
            structure_seg1, structure_seg2 = image1, image2
            structure_count1, structure_count2 = len(np.unique(structure_seg1)), len(np.unique(structure_seg2))
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
        im1_mapping = self._ordered_mapping_overlaps(im1_overlap_structs)
        im2_mapping = self._ordered_mapping_overlaps(im2_overlap_structs)
        '''io.imshow(np.amax(np.stack([structure_seg1*overlap_image, structure_seg2*overlap_image, np.zeros_like(overlap_image)], axis=-1), axis=0))
        plt.show()'''
        overlap_pair_volume_shared = np.zeros(tuple([len(im1_overlap_structs), len(im2_overlap_structs)]))
        for over_regions in range(1, overlap_count+1):
            isolated_overlap = np.equal(overlap_regions, over_regions).astype('uint8')
            '''io.imshow(isolated_overlap)
            plt.show()'''
            image1_overlap = structure_seg1 * isolated_overlap
            image2_overlap = structure_seg2 * isolated_overlap
            image1_label, im1_volumes = np.unique(image1_overlap, return_counts=True)
            image2_label, im2_volumes = np.unique(image2_overlap, return_counts=True)
            plt.show()
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
        overlapped_structures = np.nonzero(paired_structures)  # this will return the structure pairs that overlap.
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
            ''' included_subject should contain no 0 coords since the indices mapping to struct labels and the background is ignored (kept to zero) thus
            will be zero for paired_structures[0, :] == paired_structures[:, 0] == 0 #### This should be the case but must be tested'''
            '''nonzero returns a tuple of two arrays of x-coord and y-coord. x-coord will contain multiple of the same for the different y-coord pairings.
            Any struct label not in x-coord at all means that it overlaps with no structures'''
            excluded_subj_structs = np.setdiff1d(subject_im_labels, included_subject)  # this should return the labels of the ignored structures
            excluded_ref_structs = np.setdiff1d(ref_im_labels, included_reference)
            excluded_structures = (excluded_subj_structs, excluded_ref_structs)
            '''print("Excluded structures:", excluded_subj_structs)
            print("Subject image structure label matches?")
            print(subject_im_labels)
            print(subject_im_labels[included_subject])
            print(included_subject)
            print("****################################################****")'''
            over_ratio = np.zeros_like(paired_structures)
            over_ratio[included_subject] += np.divide(paired_structures[included_subject].T, subject_im_volumes[included_subject]).T  # the transpose is done for broadcasting. 0 < ratio <= 1
            vol_ratio = (paired_structures > 0).astype(float)
            vol_ratio[included_subject] = (vol_ratio[included_subject].T * subject_im_volumes[included_subject]).T
            vol_ratio[:, included_reference] = (vol_ratio[:, included_reference] / ref_im_volumes[included_reference])
            '''print(np.greater(over_ratio, 0).astype(int).sum(axis=0))
            print(np.greater(over_ratio, 0).astype(int).sum(axis=1))'''
            complete_overlap = np.greater_equal(over_ratio*vol_ratio, 1)  # 1 should be max
            perfect_structs = np.nonzero(complete_overlap)[0]  # these structures are perfectly matching with the
            # print("Perfect Matches", perfect_structs)
            ''' the np.divide approach could be used but where the ratio is no transposed back to get the overlap percentage relative to the reference.
            The value of this is that the distance of the subject from the reference should also take into account the fully excluded structures on each side.
            '''
            subject_match_relations = {a: [] for a in np.unique(included_subject)}  # this will make the pair storage dictionary
            pair_wise = np.argwhere(paired_structures)
            for pw in pair_wise:
                subject_match_relations[pw[0]].append(pw[1])
            '''print("Order test:", set(list(subject_match_relations)) == set(included_subject.tolist()))
            print("Structure pairs", subject_match_relations)'''
            ''' The below will return the ratio of overlapped volume relative to the complete volume of the original/prior structure. The vol_ratio is the 
            ratio between the old image and the new image. The subject_match_relations is a dictionary for the currently overlapping structures.
            excluded_structs is a tuple with the first elem for the im1 structs not in im2 and the second elem is for the im2 structs not in im1'''
            return over_ratio, vol_ratio, subject_match_relations, excluded_structures

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

if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    system_analyst = thresholding_metrics(input_path, expert_path="C:\\RESEARCH\\Mitophagy_data\\gui params\\",
                                          auto_path="C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\CompareResults2.json")
    system_analyst.high_and_low_testing()
    # system_analyst.structure_hunting()
    # system_analyst.stack_hist_plot()
    # system_analyst.compare_thresholds_between()
    # system_analyst._structure_overlap_test()
    # print(system_analyst.exp_threshes)
    # system_analyst.analyze_low_thresholds("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\Low Threshold Metrics\\")

