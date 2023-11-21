import json

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

class AutoHystThreshold:
    def __init__(self, input_path, extension_exception=None):
        '''
        The input path can either be a singular file or a path to a folder. If a folder is provided only PNG, JPEG,
        and TIFF image files are processed by default. Provide an extension to extension_exception if other image files
        are to be processed but these cannot be guaranteed.
        :param input_path:
        :param extension_exception:
        '''
        if isfile(input_path):
            self.image_name = input_path.split("\\")[-1]
            self.singular_im = True
            self.root_path = '\\'.join(input_path.split("\\")[:-1])
        else:
            self.image_name = self.list_of_files(input_path, extension_exception)
            self.singular_im = False
            self.root_path = input_path


    def list_of_files(self, input_path, extension_exception=None):
        def file_extensions(file_name):
            if file_name.startswith('AHT_'):
                return False
            if file_name.endswith('.png'):
                return True
            if file_name.endswith('.jpg'):
                return True
            if file_name.endswith('.tif'):
                return True
            if extension_exception is not None:
                if file_name.endswith(extension_exception):
                    return True

        files_list = [f for f in listdir(input_path) if (isfile(join(input_path, f)) and file_extensions(f))]
        return files_list

    def _get_slope(self, x, y):
        if len(x) != len(y):
            print("Inconsistent x and y coordinates")
            return None, None
        else:
            slope_values = []
            for i in range(1, len(x), 1):
                slope = abs((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                slope_values.append(slope)
            new_x = x[1:]
            return slope_values, new_x

    def _flatten_list(self, nested_iter):
        new_list = []
        try:
            current_value = next(nested_iter)
            if type(current_value) is list:
                sub_iter = iter(current_value)
                new_list += self._flatten_list(sub_iter)
                resumed_result = self._flatten_list(nested_iter)
                if resumed_result is not None:
                    new_list += resumed_result
            else:
                new_list += [current_value]
                next_value = self._flatten_list(nested_iter)
                if next_value is None:
                    return new_list
                else:
                    return new_list + next_value
        except StopIteration:
            return None
        return new_list

    def _moving_average(self, counts, window_size=10, rescale=False):
        adjusted = False
        if type(counts) is list and window_size > 1:
            new_list = []
            for n in range(0, int(window_size / 2)):
                new_list.append(0)
            counts = new_list + counts + new_list
            adjusted = True
        df = pd.DataFrame(counts)
        moving_average = df.rolling(window_size, center=True)
        moving_average = moving_average.mean()
        average_results = self._flatten_list(iter(moving_average.values.tolist()))
        if adjusted:
            window_offset = int(window_size / 2)
            average_results = average_results[window_offset:-window_offset]
            if rescale:
                for i in range(1, window_offset + 1):
                    average_results[i - 1] = (average_results[i - 1] * 10) / i
                    average_results[-i] = (average_results[-i] * 10) / i
        return average_results

    def _invert_rescaler(self, values):
        if max(values) == 0:
            inverted = np.array([1 for inv in range(len(values))])
            inverted_dict = {values[inv]: 1 for inv in range(len(values))}
        else:
            inverted = np.array([(max(values) - values[inv]) / max(values) for inv in range(len(values))])
            inverted_dict = {values[inv]: inverted[inv] for inv in range(len(values))}
        return inverted, inverted_dict

    def _grayscale(self, image):
        image_shape = image.shape
        if len(image_shape) > 3:
            if image_shape[-1] == 3 and image_shape[-1] != image_shape[-2] and image_shape[-2] == image_shape[-3]:
                return np.mean(image, axis=-1)
        return image

    def _testing_knee(self, img, cutoff=1, log_hist=False, sensitivity=1):
        counts, centers = histogram(img, nbins=256)
        if cutoff < centers[0]:
            cut = 1
        else:
            cut = np.where(centers == cutoff)[0][0]
        counts = counts[cut:]
        centers = centers[cut:]
        if log_hist:
            counts = np.log10(counts+1)
        first_knee = int(KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing", S=sensitivity).knee)
        return first_knee

    def _low_select(self, img):
        '''
        This function selects the low threshold for an image using the Kneedle algorithm.
        :param img:
        :return: chosen_knee, valid
        '''
        normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2)
        log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2)
        otsu_thresh = threshold_otsu(img)
        '''print("Normal Knee:", normal_knee, "Log Knee:", log_knee, "Otsu Thresh:", otsu_thresh, "Triangle:", threshold_triangle(img),
              "Mean:", threshold_mean(img), "Yen", threshold_yen(img), "Li", threshold_li(img))'''
        valid = True
        if otsu_thresh <= normal_knee:
            chosen_knee = normal_knee
        elif otsu_thresh > normal_knee and normal_knee > log_knee:
            chosen_knee = normal_knee
        else:
            chosen_knee = log_knee
        if log_knee <= threshold_triangle(img):
            valid = False
        if not valid:
            cut = int(threshold_mean(img))
            normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2, cutoff=cut)
            log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2, cutoff=cut)
            valid = True
            if normal_knee >= log_knee:
                chosen_knee = normal_knee
            else:
                if otsu_thresh <= normal_knee:
                    chosen_knee = normal_knee
                else:
                    chosen_knee = log_knee
        return chosen_knee, valid

    def _ihh_get_best(self, image, low_thresh=None, testing=False, test_distrib=False):
        '''
        This is taken from SystemAnalysis. This needs to be adjusted to generate the IHH of an input image
        and return the intensities plus counts. If the low_thresh parameter is None then will return a low threshold
        else if a low threshold is provided then only the IHH details will be retained. Intensities and voxel counts
        will be in ascending order
        :param image: Image for which an IHH will be determined
        :param low_thresh: Optional value. This can be provided so that the low threshold does not need to be calculated
        again.
        :return: intens, voxels(, low thresh) The intensities and voxel counts of the image IHH with the low thresh
        being returned optionally. A low thresh will be returned if not provided in the argument.
        '''
        image = self._grayscale(image)
        lw_thrsh = low_thresh if low_thresh is not None else self._low_select(img=image)[0]
        mask_low = image > lw_thrsh
        labels_low, num_labels = ndi.label(mask_low)
        valid_structures = np.stack([labels_low, image*(mask_low.astype('int'))], axis=-1) # The two labels have been stacked
        valid_structures = np.reshape(valid_structures, (-1, valid_structures.shape[-1])) # The dual label image has been flattened save for the label pairs
        sort_indices = np.argsort(valid_structures[:, 0])
        valid_structures = valid_structures[sort_indices]
        label_set, start_index, label_count = np.unique(valid_structures[:, 0], return_index=True, return_counts=True)
        end_index = start_index + label_count
        max_labels = np.zeros(tuple([len(label_set), 3]))

        for t in range(len(label_set)):
            max_labels[t, 0] = label_set[t]
            max_labels[t, 1] = valid_structures[slice(start_index[t], end_index[t]), 1].max()
            max_labels[t, 2] = label_count[t]
        intensity_index = max_labels[:, 1].argsort() # intensity index. use [::-1] to reverse
        voxels_sorted = max_labels[intensity_index, 2][1:].astype(int)
        intensity_mapping = max_labels[intensity_index, 1][1:].astype(int) # This will map the actual intensity values
        intensity_values, start_positions, sizes = np.unique(intensity_mapping, return_counts=True, return_index=True)
        consolidated_intensities = np.zeros_like(intensity_values)
        for t in range(0, len(start_positions)):
            index_back = start_positions[t]+sizes[t]
            consolidated_intensities[t] = voxels_sorted[start_positions[t]:index_back].sum()


        full_intensity_range = np.arange(0, 256)
        ihh_range = np.zeros_like(full_intensity_range)
        ihh_range[intensity_values] = consolidated_intensities
        ihh_range = np.cumsum(ihh_range[intensity_values.min():][::-1])[::-1]
        full_intensity_range = full_intensity_range[intensity_values.min():]

        def build_ihh_2():
            '''This is designed to flip the voxel and intensity distributions, then iterate across the
            reverse with the cumulative sum and then flip it back afterwards. This way even if there are only
            a few intensity points there will still be a distribution between the low thresh and the maximum intensity.
            Will use full_intensity_range for the intensities and ihh_range for the voxels.
            '''
            max_intens = full_intensity_range.max()
            bottom_intens = lw_thrsh + 1
            intens_array = np.linspace(bottom_intens, max_intens, num=max_intens+1-bottom_intens)
            index_array = full_intensity_range - bottom_intens
            voxel_canvas = np.zeros_like(intens_array)
            voxel_canvas[index_array] = ihh_range
            diminishing_cumulative_counts = np.flip(np.cumsum(np.flip(voxel_canvas))) #the second flip is to reorient it
            return intens_array, diminishing_cumulative_counts

        if test_distrib:
            intens, voxels = build_ihh_2()
        else:
            intens, voxels = full_intensity_range, ihh_range

        if low_thresh is not None:
            return intens, voxels

        return intens, voxels, lw_thrsh

    def inverted_thresholding_final(self, image, voxel_bias=True, window_option=None, testing_ihh=False):
        '''
        This version will include everything and is the final version for complete thresholding
        :param image:
        :param voxel_bias:
        :param window_option:
        :return: high_thresh, low_thresh
        '''

        low_thresh, valid_low = self._low_select(image)
        intens, voxels = self._ihh_get_best(image, low_thresh, test_distrib=True)
        #I flip in _ihh_get_best for some reason. This will be remedied in future
        if intens.shape[0] == 1:
            return intens[0]-1, low_thresh
        slopes, slope_points = self._get_slope(intens, voxels)
        mving_slopes = self._moving_average(slopes, window_size=8)
        inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
        voxel_weights = voxels/voxels.max()

        def numerator(span):
            num = np.multiply(inverted_slopes[:span], intens[:span])
            if voxel_bias:
                num = np.multiply(num, voxel_weights[:span])
            return np.sum(num)

        def denominator(span):
            denom = inverted_slopes[:span]
            if voxel_bias:
                denom = np.multiply(denom, voxel_weights[:span])
            return np.sum(denom)

        def get_mass(span):
            if voxel_bias:
                ratio = np.sum(np.multiply(inverted_slopes[:span], voxel_weights[:span]))
                if inverted_slopes.shape[0] != voxel_weights.shape[0]:
                    ratio = ratio/np.sum(np.multiply(inverted_slopes, voxel_weights[:-1]))
                else:
                    ratio = ratio / np.sum(np.multiply(inverted_slopes, voxel_weights))
                return ratio

            ratio = np.sum(inverted_slopes[:span])
            ratio = ratio/np.sum(inverted_slopes)

            return ratio

        if window_option is None:
            return numerator(len(inverted_slopes))/len(inverted_slopes), low_thresh
        elif window_option == 0:
            return numerator(len(inverted_slopes)) / denominator(len(inverted_slopes)), low_thresh
        elif window_option == 1:
            cumulative_centroid = 0
            cumulative_window_weight = 0
            for t in range(len(inverted_slopes), 0, -1):
                window_weight = t/len(inverted_slopes)
                if denominator(t) != 0:
                    cumulative_centroid += (numerator(t)/denominator(t))*window_weight
                    cumulative_window_weight += window_weight
            cumulative_window_weight = 1 if cumulative_window_weight == 0 else cumulative_window_weight
            return cumulative_centroid/cumulative_window_weight, low_thresh
        elif window_option == 2:
            cumulative_centroid = 0
            cumulative_window_weight = 0
            for t in range(len(inverted_slopes), 0, -1):
                window_weight = get_mass(t)
                cumulative_centroid += (numerator(t)/denominator(t))*window_weight
                cumulative_window_weight += window_weight
            return cumulative_centroid / cumulative_window_weight, low_thresh

    def threshold_image(self, image, low_thresh, high_thresh):
        if high_thresh <= low_thresh:
            print("High thresh too low:", high_thresh)
            high_thresh = high_thresh + low_thresh
        thresholded_image = apply_hysteresis_threshold(image, low_thresh, high_thresh).astype("uint8")
        return thresholded_image

    def run(self, output_path=None, ihh_bias=0, window_bias=0):
        '''
        This will run the AHT thresholding for the images provided in the class object. The ihh_bias designates the
        IHH bias for the run (with 0 == no IHH bias; 1 == IHH bias applied). The Window bias is defaulted to 0
        with window_bias == 0 for no Window bias applied; window_bias == 1 for the Window width bias applied;
        and window_bias == 2 for the Window mass bias applied. The output_path designates the save location of the
        result and if left None then it will be saved in the source location with the added prefix AHT_
        :param output_path:
        :param ihh_bias:
        :param window_bias:
        :return:
        '''
        if self.singular_im:
            image = io.imread(join(self.root_path, self.image_name))
            high_thresh, low_thresh = self.inverted_thresholding_final(image, ihh_bias, window_bias)
            print(high_thresh)
            thresholded_image = self.threshold_image(image, low_thresh, high_thresh)*255
            if output_path is None:
                save_name = join(self.root_path, 'AHT_' + self.image_name)
                io.imsave(save_name, thresholded_image)
            else:
                save_name = join(output_path, 'AHT_' + self.image_name)
                io.imsave(save_name, thresholded_image)
        else:
            for im_name in self.image_name:
                image = io.imread(join(self.root_path, im_name))
                high_thresh, low_thresh = self.inverted_thresholding_final(image, ihh_bias, window_bias)
                thresholded_image = self.threshold_image(image, low_thresh, high_thresh) * 255
                if output_path is None:
                    save_name = join(self.root_path, 'AHT_' + im_name)
                    io.imsave(save_name, thresholded_image)
                else:
                    save_name = join(output_path, 'AHT_' + im_name)
                    io.imsave(save_name, thresholded_image)

if __name__ == "__main__":
    input_path = ""
    output_path = None
    aht_thresholder = AutoHystThreshold(input_path)
    aht_thresholder.run(output_path)