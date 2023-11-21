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

from CleanThresholder import AutoThresholder

class imported_thresholder(AutoThresholder):

    def __init__(self, input_paths, expert_path=None, deconv_paths=None):
        AutoThresholder.__init__(self, input_paths, deconv_paths)
        self.file_list_index = 0
        self.current_prepared_image = self._prepare_image()
        self.current_low_thresh = self._get_low_thresh()
        self.expert_parameters = None
        if expert_path is not None:
            self._import_experts(expert_path)

    def _import_experts(self, exp_path):
        '''
        To do:
        1. Import the list of expert threshold params for a file
        2. Order by sample name and store per expert
        :param expert_path:
        :return:
        '''
        expert_files = [exp for exp in listdir(exp_path) if exp.endswith("_thresholds.json")]
        expert_label = ["A", "B", "C", "D"]
        expert_names = {"A": None, "B": None, "C":None, "D":None}
        expert_dict = {}
        counter = 0
        for exp_f in expert_files:
            expert_names[expert_label[counter]] = exp_f.split("_")[0]
            with open(exp_path + exp_f, "r") as j:
                expert_info = json.load(j)
            for k, v in expert_info.items():
                if k not in expert_dict:
                    expert_dict[k] = {}
                expert_dict[k][expert_label[counter]] = v
            counter += 1
        self.expert_parameters = expert_dict

    def _prepare_image(self):
        '''
        Prepares the image by grayscaling and separating timeframe but since timeframes are not used in testing
        an index of [0] is applied to image_set
        :param indexed_file:
        :return:
        '''
        image = io.imread(self.file_list[self.file_list_index][0])
        gray_image = self._grayscale(image)
        image_set = self._timeframe_sep(gray_image, self.file_list[self.file_list_index][1])[0]
        return image_set

    def _get_low_thresh(self):
        '''
        This is just to get the low threshold. Valid_low will not be used but will be inspected if False
        :return:
        '''
        low_thresh, valid_low = self._low_select(self.current_prepared_image)
        return low_thresh

    def _next_image(self):
        '''
        This will iterate it through to the next file. If it has reached the end of the file then it will terminate
        :return:
        '''
        self.file_list_index += 1
        if self.file_list_index < len(self.file_list):
            self.current_prepared_image = self._prepare_image()
            self.current_low_thresh = self._get_low_thresh()
            return True
        else:
            return False


    def centroid_tests(self, intensities, smoothed_ihh, ihh_values=None, version=1):
        '''
        This will calculate the centroid values for the gradient IHH. Version 0 is for the naive centroid and Version 1
        is for the robust centroid. If the ihh_values is assigned None then no IHH weighting will be applied.
        Regardless, a version with the IHH weights applied to the denominator is also calculated. No IHH weighting
        returns a single scalar while with IHH weighting 2 values are returned for Naive centroid and 3 for Robust
        centroid.
        :param intensities: The intensity values of the IHH
        :param smoothed_ihh: The inverted and smoothed IHH gradient distribution
        :param ihh_values: The original IHH values
        :param version: The selector for naive and robust centroid (0 or 1)
        :return:
        '''

        if ihh_values is not None:
            if type(ihh_values) is list:
                ihh_values = np.array(ihh_values)
            elif type(ihh_values) is not np.ndarray:
                print("This is not a valid list or numpy array")
                ihh_values = None
            if ihh_values.max() > 1:
                #This will normalize the IHH weights
                ihh_values = ihh_values/ihh_values.max()

        weighted_intensities = np.multiply(smoothed_ihh, intensities)

        def get_numerator(weighted=False):
            if weighted:
                if ihh_values is None:
                    raise Exception("There are no IHH values prodided yet the IHH weighting has been requested")
                return np.sum(np.multiply(weighted_intensities, ihh_values))
            else:
                return np.sum(weighted_intensities)

        if version == 0:
            #This will return the 'bad centroid' which does not use the sum of weights in the denom
            denominator = len(weighted_intensities)
            if ihh_values is None:
                return [get_numerator()/denominator]
            else:
                results = [get_numerator()/denominator, get_numerator(True)/denominator]
                return results
        else:
            denominator = smoothed_ihh
            if ihh_values is None:
                return [get_numerator()/np.sum(denominator)]
            else:
                results = [get_numerator()/np.sum(denominator), get_numerator(True)/np.sum(denominator),
                           get_numerator(True)/np.sum(np.multiply(denominator, ihh_values))]
                return results

        # 3. Need to hook in expert results to compare image diffs and use the same low threshold


    def ihh_weighting_test(self, external_low_thresh=None, win_size=8, version=1):
        '''
        This function will calculate the
        Need to add IHH weighting flag
        :param external_low_thresh:
        :param win_size:
        :param version:
        :return:
        '''
        if external_low_thresh is None:
            intens, voxels = self._ihh_get_best(self.current_prepared_image, self.current_low_thresh)
            lw_thresh = self.current_low_thresh
        else:
            intens, voxels = self._ihh_get_best(self.current_prepared_image, external_low_thresh)
            lw_thresh = external_low_thresh

        intens = intens[:-1]
        voxels = voxels[:-1]
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        centroid_results = self.centroid_tests(intens[:-1], inverted_slopes, ihh_values=voxels[:-1])
        print(centroid_results)
        '''sns.lineplot(x=intens[:-1], y=inverted_slopes)
        for t in range(0, len(centroid_results)):
            colour_range = ['k', 'r', 'g']
            plt.axvline(x=centroid_results[t], color=colour_range[t])
        plt.show()'''
        return intens[:-1], inverted_slopes, centroid_results

    def _centroid_compare(self, save_path=None, show_plots=False):
        '''
        This function will iterate through all of the Experts for the current sample. The current file index needs to
        be iterated to check the next sample.
        :return:
        '''

        current_file_name = self.file_list[self.file_list_index][1]
        expert_data = self.expert_parameters[current_file_name]
        threshold_compare = {"Exp":[], "Low":[], "Exp_High":[], "Auto_High":[]}
        for k, v in expert_data.items():
            curr_low = v['low']
            intensities, voxels, centroid_results = self.ihh_weighting_test(curr_low)
            threshold_compare["Exp"].append(k)
            threshold_compare["Low"].append(curr_low)
            threshold_compare["Exp_High"].append(v['high'])
            threshold_compare["Auto_High"].append({"No W":centroid_results[0], "Top W":centroid_results[1],
                                                   "Both W":centroid_results[2]})

        def overlay_images(low, high, auto_highs):
            expert_im = np.amax(apply_hysteresis_threshold(self.current_prepared_image, low, high).astype(int), axis=0)
            auto_im_list = []
            print("Auto", auto_highs)
            for t, d in auto_highs.items():
                auto_im = np.amax(apply_hysteresis_threshold(self.current_prepared_image, low, d).astype(int), axis=0)
                auto_im_list.append(auto_im)
            rgb_layered = np.stack(auto_im_list, axis=-1)*255
            return expert_im, rgb_layered

        print("Figure 1")
        print("Expert", threshold_compare["Low"][0], threshold_compare["Exp_High"][0])
        fig1, axs = plt.subplots(1, 2)
        exp, layer = overlay_images(threshold_compare["Low"][0], threshold_compare["Exp_High"][0], threshold_compare["Auto_High"][0])
        axs[0].imshow(exp)
        axs[1].imshow(layer)
        print("Figure 2")
        print("Expert", threshold_compare["Low"][1], threshold_compare["Exp_High"][1])
        fig2, axs = plt.subplots(1, 2)
        exp, layer = overlay_images(threshold_compare["Low"][1], threshold_compare["Exp_High"][1],
                                      threshold_compare["Auto_High"][1])
        axs[0].imshow(exp)
        axs[1].imshow(layer)
        print("Figure 3")
        print("Expert", threshold_compare["Low"][2], threshold_compare["Exp_High"][2])
        fig3, axs = plt.subplots(1, 2)
        exp, layer = overlay_images(threshold_compare["Low"][2], threshold_compare["Exp_High"][2],
                                      threshold_compare["Auto_High"][2])
        axs[0].imshow(exp)
        axs[1].imshow(layer)
        print("Figure 4")
        print("Expert", threshold_compare["Low"][3], threshold_compare["Exp_High"][3])
        fig4, axs = plt.subplots(1, 2)
        exp, layer = overlay_images(threshold_compare["Low"][3], threshold_compare["Exp_High"][3],
                                      threshold_compare["Auto_High"][3])
        axs[0].imshow(exp)
        axs[1].imshow(layer)

        if show_plots:
            plt.show()

        if save_path is not None:
            save_name = current_file_name.split(".")[0]
            fig1.savefig(join(save_path, save_name + "_Exp1.png"))
            fig2.savefig(join(save_path, save_name + "_Exp2.png"))
            fig3.savefig(join(save_path, save_name + "_Exp3.png"))
            fig4.savefig(join(save_path, save_name + "_Exp4.png"))
            with open(join(save_path, save_name + ".json"), "w") as j:
                json.dump(threshold_compare, j)


    def iterate_through_samples(self, save_point):
        for t in range(len(self.file_list)):
            print("Sample:", self.file_list[self.file_list_index][1])
            self._centroid_compare(save_path=save_point)
            self._next_image()

    def struct_highlighting(self, image):
        '''
        This function will get use the image name to select the image from self.im_paths and then get the max intensity
        label for each structure. This will result in an image where for each structure there will be a singular
        intensity value for each composing voxel.
        :param image_name:
        :return:
        '''
        low_thresh, bleh = self._low_select(image)  # bleh is used for the boolian as it is unrequired
        low_mask = image > low_thresh
        labels_low, num_labels = ndi.label(low_mask)
        valid_structures = np.stack([labels_low, image*(low_mask.astype('int'))], axis=-1)
        valid_structures = np.reshape(valid_structures, (-1, valid_structures.shape[-1]))
        valid_structures = valid_structures[np.argsort(valid_structures[:, 0])]
        label_set, start_index, label_count = np.unique(valid_structures[:, 0], return_index=True, return_counts=True)
        end_index = start_index + label_count
        struct_max_intensities = np.zeros(tuple([len(label_set), 2]))
        canvas_im = np.zeros_like(labels_low)
        for t in range(len(label_set)):
            canvas_im[np.equal(labels_low, label_set[t])] = valid_structures[slice(start_index[t], end_index[t]), 1].max()
        return canvas_im

    def get_centroid_compare(self, image_name):
        print(image_name)
        image = io.imread(self.image_paths[image_name])
        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, False)
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        voxel_weights = voxels / voxels.max()

        def no_voxel_centroid():
            num = np.sum(np.multiply(inverted_slopes, intens[:-1]))
            denom = np.sum(inverted_slopes)
            return num/denom

        def voxel_only_centroid():
            num = np.multiply(intens, voxel_weights)
            denom = voxel_weights
            return np.sum(num)/np.sum(denom)

        def voxel_centroid():
            num = np.sum(np.multiply(np.multiply(inverted_slopes, voxel_weights[:-1]), intens[:-1]))
            denom = np.sum(np.multiply(inverted_slopes, voxel_weights[:-1]))
            return num/denom

        normal_centroid = no_voxel_centroid()
        ihh_centroid = voxel_only_centroid()
        both_centroid = voxel_centroid()
        print(intens.shape, slope_points.shape, inverted_slopes.shape)
        plt.rcParams.update({'font.size': 18})
        sns.lineplot(x=intens[:-1], y=inverted_slopes, color='r', label='Inverted\nGradient\nDistribution')
        sns.lineplot(x=intens, y=voxel_weights, color='g', label='Normalized\nIHH\nDistribution')
        plt.axvline(x=normal_centroid, color='r', dashes=[4, 4], label='Centroids')
        plt.axvline(x=ihh_centroid, color='g', dashes=[4, 4])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legend_handles[2].set_color('k')
        plt.ylabel('Magnitude')
        plt.xlabel('High Threshold Intensities')
        plt.show()
        sns.lineplot(x=intens[:-1], y=np.multiply(inverted_slopes, voxel_weights[:-1]), color='k')
        plt.axvline(x=both_centroid, label='Centroid', color='k', dashes=[4, 4])
        plt.ylabel('Magnitude')
        plt.xlabel('High Threshold Intensities')
        plt.title("Inverted Gradient Distribution with IHH Biasing applied")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        plt.show()

    def get_mip_highlights(self, image_name):
        '''
        This function will get the MIP
        :param image_name:
        :return:
        '''
        print(image_name)
        image = io.imread(self.image_paths[image_name])
        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, False)
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        voxel_weights = voxels / voxels.max()
        highlight_image = self.struct_highlighting(image)

        def no_voxel_centroid():
            num = np.sum(np.multiply(inverted_slopes, intens[:-1]))
            denom = np.sum(inverted_slopes)
            return num / denom

        def voxel_only_centroid():
            num = np.multiply(intens, voxel_weights)
            denom = voxel_weights
            return np.sum(num) / np.sum(denom)

        def voxel_centroid():
            num = np.sum(np.multiply(np.multiply(inverted_slopes, voxel_weights[:-1]), intens[:-1]))
            denom = np.sum(np.multiply(inverted_slopes, voxel_weights[:-1]))
            return num / denom

        inverted_im = np.amax((highlight_image > no_voxel_centroid()).astype(int)*highlight_image, axis=0).astype('uint8')
        voxel_im = np.amax((highlight_image > voxel_only_centroid()).astype(int)*highlight_image, axis=0).astype('uint8')
        both_im = np.amax((highlight_image > voxel_centroid()).astype(int)*highlight_image, axis=0).astype('uint8')

        blank_im = np.zeros_like(voxel_im)
        rg_stack = np.stack([inverted_im, voxel_im, both_im], axis=-1)
        raw_mip = np.amax(image, axis=0).astype('uint8')
        highlight_mip = np.amax(highlight_image, axis=0).astype('uint8')
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\raw_mip.png", np.stack([raw_mip, raw_mip, raw_mip], axis=-1))
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\highlight_mip.png",highlight_mip, cmap='viridis')
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\rg_mip.png",rg_stack)
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\weighted_mip.png",both_im)

    def centroid_compare(self, image_name):
        print(image_name)
        image = io.imread(self.image_paths[image_name])
        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, False)
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        voxel_weights = voxels / voxels.max()
        highlight_image = self.struct_highlighting(image)

        def numerator():
            num = np.multiply(inverted_slopes, intens[:-1])
            return np.sum(num)

        def denominator():
            denom = inverted_slopes
            return np.sum(denom)

        incorrect_centroid = int(numerator()/len(inverted_slopes))
        correct_centroid = int(numerator()/denominator())

        plt.rcParams.update({'font.size': 22})
        sns.lineplot(x=intens[:-1], y=inverted_slopes, color='k')
        plt.axvline(x=incorrect_centroid, color='g', dashes=[4,4], label='Incorrect\nCentroid')
        plt.axvline(x=correct_centroid, color='r', dashes=[4,4], label='Correct\nCentroid')
        plt.ylabel('Magnitude')
        plt.xlabel('High Threshold Intensities')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        plt.show()

        '''above_low = np.amax((image > low_thresh).astype(int), axis=0)
        incorrect_centr = np.amax(apply_hysteresis_threshold(image, low_thresh, incorrect_centroid), axis=0)
        correct_centr = np.amax(apply_hysteresis_threshold(image, low_thresh, correct_centroid), axis=0)

        combined = np.stack([correct_centr, incorrect_centr, above_low], axis=-1).astype("uint8")*255
        io.imshow(combined)
        plt.show()
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\centroid_compare.png", combined)'''

    def weighting_mip(self, image_name):
        '''
        This function will get the weighting MIP where it will evaluate
        :param image_name:
        :return:
        '''
        print(image_name)
        print(self.image_paths)
        image = io.imread(self.image_paths[image_name])
        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, False)
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        voxel_weights = (voxels / voxels.max())[:len(inverted_slopes)]

        def numerator(span, voxel_bias=False):
            num = np.multiply(inverted_slopes[:span], intens[:span])
            if voxel_bias:
                num = np.multiply(num, voxel_weights[:span])
            return np.sum(num)

        def denominator(span, voxel_bias=False):
            denom = inverted_slopes[:span]
            if voxel_bias:
                denom = np.multiply(denom, voxel_weights[:span])
            return np.sum(denom)

        def get_mass(span, voxel_bias=False):
            if voxel_bias:
                ratio = np.sum(np.multiply(inverted_slopes[:span], voxel_weights[:span]))
                ratio = ratio/np.sum(np.multiply(inverted_slopes, voxel_weights))
                return ratio

            ratio = np.sum(inverted_slopes[:span])
            ratio = ratio/np.sum(inverted_slopes)

            return ratio

        highlight_image = self.struct_highlighting(image)

        def thresh_highlight(thresh):
            return np.amax((highlight_image > thresh).astype(int)*highlight_image, axis=0).astype('uint8')


        cumulative_centroid_1 = 0
        cumulative_window_weight_1 = 0
        cumulative_centroid_2 = 0
        cumulative_window_weight_2 = 0
        for t in range(len(inverted_slopes), 0, -1):
            window_weight_1 = t / len(inverted_slopes)
            if denominator(t) != 0:
                cumulative_centroid_1 += (numerator(t) / denominator(t)) * window_weight_1
                cumulative_window_weight_1 += window_weight_1
            window_weight_2 = get_mass(t)
            cumulative_centroid_2 += (numerator(t) / denominator(t)) * window_weight_2
            cumulative_window_weight_2 += window_weight_2

        centroid_a = int(cumulative_centroid_1 / cumulative_window_weight_1)
        centroid_b = int(cumulative_centroid_2 / cumulative_window_weight_2)
        no_window_centroid = int(numerator(len(inverted_slopes)) / denominator(len(inverted_slopes)))

        print(no_window_centroid, centroid_a, centroid_b)
        layering_one = np.stack([np.amax(highlight_image, axis=0).astype('uint8')])
        #with voxel bias applied

        cumulative_centroid_1 = 0
        cumulative_window_weight_1 = 0
        cumulative_centroid_2 = 0
        cumulative_window_weight_2 = 0
        for t in range(len(inverted_slopes), 0, -1):
            window_weight_1 = t / len(inverted_slopes)
            if denominator(t) != 0:
                cumulative_centroid_1 += (numerator(t, True) / denominator(t, True)) * window_weight_1
                cumulative_window_weight_1 += window_weight_1
            window_weight_2 = get_mass(t, True)
            cumulative_centroid_2 += (numerator(t, True) / denominator(t, True)) * window_weight_2
            cumulative_window_weight_2 += window_weight_2

        centroid_a = int(cumulative_centroid_1 / cumulative_window_weight_1)
        centroid_b = int(cumulative_centroid_2 / cumulative_window_weight_2)
        no_window_centroid = int(numerator(len(inverted_slopes), True) / denominator(len(inverted_slopes), True))

        print("With voxel bias")
        print(no_window_centroid, centroid_a, centroid_b)



        '''highlight_image = self.struct_highlighting(image)
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\raw_highlight.png",
                  np.amax(highlight_image, axis=0).astype('uint8'), cmap='viridis')
        thresh_image = ((highlight_image > no_window_centroid).astype(int)*highlight_image).astype('uint8')
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\zero_window.png",
                  np.amax(thresh_image, axis=0).astype('uint8'), cmap='viridis')
        thresh_image = ((highlight_image > centroid_a).astype(int) * highlight_image).astype('uint8')
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\window_width.png",
                  np.amax(thresh_image, axis=0).astype('uint8'), cmap='viridis')
        thresh_image = ((highlight_image > centroid_b).astype(int) * highlight_image).astype('uint8')
        io.imsave("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Chapter 3 Figs\\window_mass.png",
                  np.amax(thresh_image, axis=0).astype('uint8'), cmap='viridis')'''

    def annotate_graph(self, image_name, percent):
        print(image_name)
        print(self.image_paths)
        image = io.imread(self.image_paths[image_name])
        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, False)
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)

        def numerator(span):
            num = np.multiply(inverted_slopes[:span], intens[:span])
            return np.sum(num)

        def denominator(span):
            denom = inverted_slopes[:span]
            return np.sum(denom)

        def subscripting(sub):
            normal = "1234"
            sub_s = "₁₂₃₄"
            sub.translate(sub.maketrans(''.join(normal), ''.join(sub_s)))

        norm_centroid = numerator(len(inverted_slopes))/denominator(len(inverted_slopes))
        colour_range = ['red', 'green', 'blue']
        sub_list = ['₁', '₂', '₃', '₄']
        plt.rcParams.update({'font.size': 12})
        offset = 1.4
        if type(percent) is not list:
            percent = [percent]
        for t in range(len(percent)):
            mid_arrow = int((intens[-1] - intens[0]) * percent[t]) + intens[0]
            upper_bound = mid_arrow - intens[0]
            sns.lineplot(x=intens[:upper_bound], y=inverted_slopes[:upper_bound], color='k')
            '''plt.axvline(x=norm_centroid, dashes=[4, 2], color='k')
            plt.annotate('C', xy=(norm_centroid - 1, 1.05),
                         xytext=(norm_centroid, 1.05), annotation_clip=False)'''
            plt.ylabel('Magnitude')
            plt.xlabel('Intensities')
            bottom_arrow = intens[0]
            #top_arrow = intens[-2]
            lower_window_mid = int((mid_arrow - bottom_arrow) / 2) + intens[0]
            #upper_window_mid = int((top_arrow - mid_arrow) / 2) + mid_arrow
            win_centroid = numerator(int((intens[-1] - intens[0]) * percent[t]))/denominator(int((intens[-1] - intens[0]) * percent[t]))
            min_value, max_value = inverted_slopes[:upper_bound].min(), inverted_slopes[:upper_bound].max()
            plt_scale = (max_value - min_value)*0.05
            plot_min = min_value - plt_scale - plt_scale*offset
            plt.annotate('', xy=(bottom_arrow, plot_min), xytext=(mid_arrow, plot_min), arrowprops=dict(arrowstyle='<->', color=colour_range[t]),
                         annotation_clip=False)
            plt.annotate('', xy=(bottom_arrow, plot_min), xytext=(mid_arrow, plot_min),
                         arrowprops=dict(arrowstyle='|-|', color=colour_range[t]),
                         annotation_clip=False)

            '''plt.annotate('', xy=(mid_arrow, -0.1), xytext=(top_arrow, -0.1),
                         arrowprops=dict(arrowstyle='<->', color='green'),
                         annotation_clip=False)
            plt.annotate('', xy=(mid_arrow, -0.1), xytext=(top_arrow, -0.1),
                         arrowprops=dict(arrowstyle='|-|', color='green'),
                         annotation_clip=False)
            plt.annotate('X-Window', xy=(upper_window_mid-4, -0.15), xytext=(upper_window_mid-4, -0.15), annotation_clip=False)'''
            plt.axvline(x=win_centroid, dashes=[4,4], color=colour_range[t])
            plt.annotate('C{}'.format(sub_list[t]), xy=(win_centroid-2, max_value + plt_scale*1.02),
                         xytext=(win_centroid-2, max_value + plt_scale*1.02), annotation_clip=False)
            plt.annotate('W{}'.format(sub_list[t]), xy=(lower_window_mid-2, plot_min-0.55*plt_scale), xytext=(lower_window_mid-2, plot_min-0.55*plt_scale),
                         annotation_clip=False)
            '''plt.annotate('W{}'.format(sub_list[t]), xy=(lower_window_mid-2, -0.11-0.038*t-offset), xytext=(lower_window_mid-2, -0.11-0.038*t-offset),
                         annotation_clip=False)'''
            plt.show()





if __name__ == "__main__":
    input_path = ["C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"]
    expert_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\"
    save_point = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\voxel_bias_testing\\"
    tester = imported_thresholder(input_paths=input_path, expert_path=expert_path)
    #tester.annotate_graph("CCCP_1C=1T=0.tif", [0.8, 0.6, 0.4])
    tester.get_mip_highlights("CCCP_1C=1T=0.tif")
    #tester.centroid_compare("CCCP_1C=1T=0.tif")
    #tester.weighting_mip("CCCP_2C=0T=0.tif")
    #tester.weighting_mip("CCCP_1C=1T=0.tif")
    #tester.weighting_mip("CCCP+Baf_2C=1T=0.tif")
    #tester.get_centroid_compare("CCCP_1C=1T=0.tif")
    #tester.get_centroid_compare("CCCP+Baf_2C=1T=0.tif")
    #tester.iterate_through_samples(save_point=save_point)
    #tester.ihh_weighting_test()
