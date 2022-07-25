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
import time
from scipy import ndimage as ndi

class AutoThresholder:
    def __init__(self, input_paths, deconvolved_paths=None):
        self.input_path = input_paths
        self.file_list = []
        self.filenames = []
        self.deconvolved_files = None
        self.deconvolved_path = deconvolved_paths
        self.image_paths = {}
        self._get_file_list()
        if deconvolved_paths is not None:
            self._get_deconvolved_files()
        self.sample_thresholds = {}
        self.distance_metrics = None

    def _get_file_list(self):
        """
        This method will extract the files along each path that are .tif files. The list will record the full path with the file name for each file in the
        first element and the name of the file as the second element. List comprehension is used if the deconvolved paths are provided else the slower loop
        is used which will collect a list of the file names.
        :return:
        """
        if self.deconvolved_path is None:
            self.file_list = [[indiv_paths + f, f] for indiv_paths in self.input_path for f in listdir(indiv_paths)
                              if isfile(join(indiv_paths, f)) and f.endswith('.tif')]
            self.image_paths = {f: indiv_paths + f for indiv_paths in self.input_path for f in listdir(indiv_paths)
                                if isfile(join(indiv_paths, f)) and f.endswith('.tif')}
        else:
            for indiv_paths in self.input_path:
                for f in listdir(indiv_paths):
                    if isfile(join(indiv_paths, f)) and f.endswith('.tif'):
                        self.file_list.append([indiv_paths + f, f])
                        self.filenames.append(f)
                        self.image_paths[f] = indiv_paths + f

    def _get_deconvolved_files(self):
        """
        This method will determine the files that are found in the deconvolution paths and provided input paths. This is purely used when the structure of
        the .tif files may have blended the timeframe and z-slice axis together erroneously. The results are stored as a dictionary with the filename as the key
        and the value being the deconvolve path with the file. This will have to be re-evaluated later as there could be identically named files between
        collections.
        :return:
        """
        self.deconvolved_files = {f: (decon_paths + f) for decon_paths in self.deconvolved_path for f in listdir(decon_paths)
                          if isfile(join(decon_paths, f)) and f.endswith('.tif') and f in self.filenames}

    def process_images(self, steepness=6, power=1):
        if len(self.sample_thresholds) == 0:
            for f in self.file_list:
                image = io.imread(f[0])
                self._acquire_thresholds(image, f[1], steepness, power)
                self.image_paths[f[1]] = f[0]
        else:
            print("Already Loaded")
            for f in self.file_list:
                self.image_paths[f[1]] = f[0]

    def _timeframe_sep(self, image, filename):
        """
        The image will be converted to grayscale prior to method argument.
        :param image:
        :param filename:
        :return:
        """
        image = image.astype('uint8')
        if image.shape[-1] == image.shape[-2]:
            # X and Y axis are identical
            if len(image.shape) > 3 or self.deconvolved_files is None:
                image = image[None, ...]
                return image
            else:
                if filename in self.deconvolved_files:
                    deconvolve_file = io.imread(self.deconvolved_files[filename])
                    deconvolve_file
                    deconvolve_shape = deconvolve_file.shape
                    if len(deconvolve_shape) > 3:
                        if deconvolve_shape[-1] != deconvolve_shape[-2] and deconvolve_shape[-1] == 3 and deconvolve_shape[-3] == deconvolve_shape[-2]:
                            deconvolve_shape = deconvolve_shape[:-1]
                    if deconvolve_shape == image.shape:
                        return [image]
                    else:
                        time_res = deconvolve_shape[0]
                        z_res = deconvolve_shape[1]
                        timeframe_shape = tuple([time_res, z_res, image.shape[-2], image.shape[-1]])
                        timeframes = np.zeros(timeframe_shape)
                        # print("Deconvolved shape", deconvolve_shape)
                        t_indexes = np.arange(time_res)
                        precise_indexes = np.array([np.arange(pi * z_res, pi * z_res + z_res) for pi in range(time_res)])
                        '''print("T index size", len(t_indexes))
                        print("Precise index size", precise_indexes.shape)'''
                        timeframes[t_indexes] = image[precise_indexes]
                        '''print("First Check:", timeframes.shape)
                        timeframes = np.zeros(timeframe_shape)
                        for t in range(time_res):
                            t_lower = t * z_res
                            t_upper = t * z_res + z_res
                        timeframes[t] = image[t_lower:t_upper]'''
                        return timeframes
                else:
                    return None

    def _acquire_thresholds(self, image, filename, steepness=6, power=1, options=[0, 1, 2]):
        print("Sample " + filename)
        gray_image = self._grayscale(image)
        print("Grayscale Image Shape:", gray_image.shape)
        image_set = self._timeframe_sep(gray_image, filename)
        print("Image is grayscale and a set")
        '''if type(steepness) is not list:
            steepness = [steepness]
        if type(power) is not list:
            power = [power]'''
        if image_set is not None:
            counter = 0
            for img in image_set:
                print("Image " + str(counter) + " is being processed")
                low_thresh, valid_low = self._low_select(img)
                print("Low thresh has been acquired")
                if valid_low:
                    iterate_start_time = time.process_time()
                    intens, voxels = self._efficient_hysteresis_iterative(img, low_thresh)
                    iterate_end_time = time.process_time()
                    print("Hysteresis Iteration took: " + str(iterate_end_time - iterate_start_time) + "s")
                    print("Cumulative Hysteresis Determined")
                    intens = intens[:-1]
                    voxels = voxels[:-1]
                    print(voxels)
                    print("Sample:", filename)
                    print("Low Threshold:", low_thresh, "image_max:", np.max(img))
                    slopes, slope_points = self._get_slope(intens, voxels)
                    mving_slopes = self._moving_average(slopes, window_size=8)
                    print("Rolling average of slopes determined")
                    inverted_threshes = {}
                    logistic_threshes = {}
                    for t in options:
                        print("Option " + str(t) + " is being processed")
                        high_invert, high_logistic = self._logist_invert_compare(mving_slopes, voxels, steepness, t)
                        inverted_threshes[t] = str(low_thresh + high_invert)
                        logistic_threshes[t] = str(low_thresh + high_logistic)
                        print("Option completed")
                    if filename not in self.sample_thresholds:
                        self.sample_thresholds[filename] = {}
                    self.sample_thresholds[filename][counter] = {"Low": str(low_thresh), "High":{"Inverted": inverted_threshes, "Logistic": logistic_threshes}}
                counter += 1

    def _specific_thresholds(self, image, excluded_type, thresh_options, steepness=6, power=1):
        low_thresh, valid_low = self._low_select(image)
        if valid_low:
            intens, voxels = self._efficient_hysteresis_iterative(image, low_thresh)
            intens = intens[:-1]
            voxels = voxels[:-1]
            slopes, slope_points = self._get_slope(intens, voxels)
            mving_slopes = self._moving_average(slopes, window_size=8)
            voxels = np.power(np.array(voxels), power)
            threshold_results = {}
            if "Inverted" not in excluded_type:
                threshold_results["Inverted"] = {}
            if "Logistic" not in excluded_type:
                threshold_results["Logistic"] = {}

            for t in thresh_options:
                if "Inverted" in threshold_results:
                    high_invert = self._inverted_thresholding(mving_slopes, voxels, t)
                    threshold_results["Inverted"][t] = [low_thresh, low_thresh + high_invert, t]
                if "Logistic" in threshold_results:
                    high_logistic = self._logistic_thresholding(mving_slopes, voxels, steepness, t)
                    threshold_results["Logistic"][t] = [low_thresh, low_thresh + high_logistic, t]
            return threshold_results
        else:
            return None


    def _generate_sigmoid(self, midpoint, k=10):
        k = k / (midpoint * 2)
        range_of_sig = []
        progress = 0
        try:
            for y in range(int(midpoint * 2) + 1):
                progress = y
                sig = 1 / (1 + math.pow(math.e, k * (y - midpoint)))
                range_of_sig.append(sig)
        except Exception as e:
            print("Error: ", e)
            print("Progress is: " + str((progress / int(midpoint * 2)) * 100) + "%")
        return range_of_sig

    def _moving_average(self, counts, window_size=10, rescale=False):
        adjusted = False
        if type(counts) is list and window_size > 1:
            new_list = []
            for n in range(0, int(window_size / 2)):
                new_list.append(0)
            counts = new_list + counts + new_list
            adjusted = True
        df = pd.DataFrame(counts)
        moving_average = df.rolling(window_size, center=True).mean()
        average_results = self._flatten_list(iter(moving_average.values.tolist()))
        if adjusted:
            window_offset = int(window_size / 2)
            average_results = average_results[window_offset:-window_offset]
            if rescale:
                print("Prior to rescaling", average_results)
                for i in range(1, window_offset + 1):
                    average_results[i - 1] = (average_results[i - 1] * 10) / i
                    average_results[-i] = (average_results[-i] * 10) / i
                print("Rescaled results", average_results)
        return average_results

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

    def _iterate_hysteresis(self, img, low_thresh):
        voxels_by_intensity, intensities = histogram(img)
        low_index = np.where(intensities == low_thresh)[0][0]
        intensities = intensities[low_index:]
        voxels_per_high_thresh = []
        template_compare_array = np.zeros_like(img)
        intensity_list = []
        for i in range(np.max(img), low_thresh, -1):
            intensity_list.append(int(i))
            threshold_result = apply_hysteresis_threshold(img, low_thresh, i).astype('uint8')
            number_of_voxels = threshold_result.sum()
            template_compare_array = np.maximum(threshold_result * i, template_compare_array)
            voxels_per_high_thresh.append(int(number_of_voxels))
        return intensity_list, voxels_per_high_thresh

    def _grayscale(self, image):
        image_shape = image.shape
        if len(image_shape) > 3:
            if image_shape[-1] == 3 and image_shape[-1] != image_shape[-2] and image_shape[-2] == image_shape[-3]:
                return np.mean(image, axis=-1)
        return image

    def _low_select(self, img):
        normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2)
        log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2)
        otsu_thresh = threshold_otsu(img)
        print("Normal Knee:", normal_knee, "Log Knee:", log_knee, "Otsu Thresh:", otsu_thresh)
        valid = True
        if otsu_thresh <= normal_knee:
            chosen_knee = normal_knee
        else:
            chosen_knee = log_knee
        if log_knee <= threshold_triangle(img):
            valid = False
        return chosen_knee, valid

    def _testing_knee(self, img, cutoff=1, log_hist=False, sensitivity=1):
        counts, centers = histogram(img, nbins=256)
        if cutoff < centers[0]:
            cut = 1
        else:
            cut = np.where(centers == cutoff)[0][0]
        counts = counts[cut:]
        centers = centers[cut:]
        if log_hist:
            counts = np.where(counts != 0, np.log10(counts), 0)

        safe_knee = True
        '''if filter_value is None:
            otsu_thresh = threshold_otsu(img)
        else:
            otsu_thresh = filter_value'''
        true_knee = 0
        knee_found = True
        first_knee = int(KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing", S=sensitivity).knee)
        '''while safe_knee:
            locator = KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing", S=sensitivity)
            knee = int(locator.knee)
            if knee > otsu_thresh and knee_found:
                true_knee = knee
                knee_found = False
                # print("True Knee", true_knee)
            if knee <= otsu_thresh:
                centers = centers[1:]
                counts = counts[1:]
            else:
                safe_knee = False'''

        return first_knee

    def _logist_invert_compare(self, ave_slopes, voxels, steepness, weighted_option=0):
        vox_max = max(voxels)
        vox_weight = [vw / vox_max for vw in voxels[:-1]]
        invert_start_time = time.process_time()
        invert_rescaled, invert_dict = self._invert_rescaler(ave_slopes)
        invert_weighted = self._apply_weights(invert_rescaled, ave_slopes)
        invert_knee_f = KneeLocator(np.linspace(0, len(invert_weighted), len(invert_weighted)), invert_weighted, S=0.1, curve="convex", direction="decreasing")
        invert_knee = int(invert_knee_f.knee)
        inverted_centroid = self._weighted_intensity_centroid(ave_slopes[invert_knee:], invert_dict, vox_weight[invert_knee:], weighted_option)
        invert_end_time = time.process_time()
        print("Inverted High Thresh Calc took: " + str(invert_end_time - invert_start_time) + "s")

        logistic_start_time = time.process_time()
        max_slope = math.ceil(max(ave_slopes))
        logist = self._generate_sigmoid(max_slope / 2, steepness)
        logist_rescaled = [logist[int(lgr)] for lgr in ave_slopes]
        logist_weighted = self._apply_weights(logist_rescaled, ave_slopes)
        logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted, S=0.1, curve="convex",
                                    direction="decreasing")
        logist_knee = int(logist_knee_f.knee)
        logist_centroid = self._weighted_intensity_centroid(ave_slopes[logist_knee:], logist, vox_weight[logist_knee:], weighted_option)
        logistic_end_time = time.process_time()
        print("Logistic High Thresh Calc took: " + str(logistic_end_time - logistic_start_time) + "s")
        return logist_centroid, inverted_centroid

    def _logistic_thresholding(self, slopes, voxels, steepness, weighted_option=0):
        vox_max = max(voxels)
        vox_weight = [vw / vox_max for vw in voxels[:-1]]
        max_slope = math.ceil(max(slopes))
        logist = self._generate_sigmoid(max_slope / 2, steepness)
        logist_rescaled = [logist[int(lgr)] for lgr in slopes]
        logist_weighted = self._apply_weights(logist_rescaled, slopes)
        logist_knee_f = KneeLocator(np.linspace(0, len(logist_weighted), len(logist_weighted)), logist_weighted, S=0.1, curve="convex",
                                    direction="decreasing")
        logist_knee = int(logist_knee_f.knee)
        logist_centroid = self._weighted_intensity_centroid(slopes[logist_knee:], logist, vox_weight[logist_knee:], weighted_option)
        return logist_centroid

    def _inverted_thresholding(self, slopes, voxels, weighted_option=0):
        vox_max = voxels.max()
        vox_weight = voxels/vox_max

        invert_rescaled, invert_dict = self._invert_rescaler(slopes)
        invert_weighted = self._apply_weights(invert_rescaled, slopes)
        invert_knee_f = KneeLocator(np.linspace(0, len(invert_weighted), len(invert_weighted)), invert_weighted, S=0.1, curve="convex", direction="decreasing")
        invert_knee = int(invert_knee_f.knee)
        inverted_centroid = self._weighted_intensity_centroid(slopes[invert_knee:], invert_dict, vox_weight[invert_knee:], weighted_option)
        return inverted_centroid

    def _weighted_intensity_centroid(self, values, weights, voxel_weights, weight_option=0):
        """
        This method will apply the weights for each value to the value. Values and voxel_weights must clip [knee:] in argument.
        The weighted window for each intensity should be calculated by applying the normalized voxel weight and the calculated weight.
        This will then be used to determine the weighted, biased centroid
        :param values:
        :param weights:
        :param voxel_weights:
        :param weight_option:
        :return:
        """
        biased_centroid = 0
        window_width_weight = 0
        #print("Values", values)
        for d in range(len(values), 0, -1):
            sum_scaled = 0
            weight_total = 0.0
            #print("Weight Total Reset")
            weight_total = float(weight_total)
            for v in range(0, d):
                val = values[v]
                if type(weights) is dict:
                    weight = weights[val]
                else:
                    weight = weights[int(val)]
                vw = voxel_weights[v]
                weight_total += weight
                #print("Total thus far:", weight_total, "Weight added:", weight)
                weighted_val = v * weight * vw
                sum_scaled += weighted_val
            if weight_option == 0:
                biased_window_centroid = sum_scaled / d
            else:
                if weight_total == 0:
                    #print("Range of weights", weights)
                    weight_total = 1
                biased_window_centroid = sum_scaled / weight_total
            biased_centroid += biased_window_centroid
            if weight_option == 2:
                window_mass = self._mass_percentage(voxel_weights, d)
                window_width_weight += window_mass
            elif weight_option == 1:
                window_width_weight += d / len(values)
            else:
                window_width_weight = len(values)
        return biased_centroid / window_width_weight

    def _mass_percentage(self, mass, cut_off):
        mass_array = np.array(mass)
        total_mass = mass_array.sum()
        mass_ratio = mass_array[:cut_off].sum()/total_mass
        return mass_ratio

    def _weighted_intensities(self, values, weights, voxel_weights):
        intensities_weighted = []
        for v in range(0, len(values)):
            val = values[v]
            if type(weights) is dict:
                weight = weights[val]
            else:
                weight = weights[int(val)]
            vw = voxel_weights[v]
            intensities_weighted.append(v * weight * vw)
        return intensities_weighted

    def _apply_weights(self, weightings, original):
        reweighted_vals = []
        for w, o in zip(weightings, original):
            reweighted_vals.append(w * o)
        return reweighted_vals

    def _invert_rescaler(self, values):
        inverted = [(max(values) - values[inv]) / max(values) for inv in range(len(values))]
        inverted_dict = {values[inv]: (max(values) - values[inv]) / max(values) for inv in range(len(values))}
        return inverted, inverted_dict

    def get_threshold_results(self):
        print(self.sample_thresholds)
        sample_list = list(self.sample_thresholds)
        sample_count = 0
        print("Samples are:")
        for s in sample_list:
            print(s)
            sample_count += len(list(self.sample_thresholds[s]))
        print("There are " + str(sample_count) + " samples.")


    def save_threshold_results(self, save_path):
        with open(save_path, 'w') as j:
            json.dump(self.sample_thresholds, j)

    def _threshold_image(self, image, low_thresh, high_thresh):
        if high_thresh <= low_thresh:
            print("High thresh too low:", high_thresh)
            high_thresh = high_thresh + low_thresh
        thresholded_image = apply_hysteresis_threshold(image, low_thresh, high_thresh).astype("int")
        return thresholded_image

    def _rgb_overlay(self, image, low_thresh, high_threshs):
        print(high_threshs)
        zero_template = np.zeros_like(image)
        image_set = []
        for i in ['0', '1', '2']:
            if i in high_threshs:
                thresholded_image = self._threshold_image(image, float(low_thresh), float(high_threshs[i]))
                image_set.append(thresholded_image)
            else:
                image_set.append(zero_template)
        rgb_overlayed = np.stack(image_set, axis=-1)
        return rgb_overlayed

    def _orig_mask_mip(self, image, low_thresh, high_thresh):
        thresholded_image = self._threshold_image(image, float(low_thresh), float(high_thresh)).astype('uint8')
        thresh_mip = (np.amax(thresholded_image, axis=0)*image.max())
        orig_mip = np.amax(image, axis=0)
        zero_layer = np.zeros_like(thresh_mip)
        orig_rgb = np.stack((orig_mip, orig_mip, orig_mip), axis=-1)
        thresh_rgb = np.stack((thresh_mip, zero_layer, zero_layer), axis=-1)
        overlayed_mip = np.clip(orig_rgb*0.65 + thresh_rgb*0.35, 0, np.max(image)).astype('uint8')
        return overlayed_mip

    def preview_thresholds_options(self, mip=False):
        self.process_images()
        for sample, stats in self.sample_thresholds.items():
            image_stack = io.imread(self.image_paths[sample])
            grayscale_stack = self._grayscale(image_stack)
            time_separated_stack = self._timeframe_sep(grayscale_stack, sample)
            for t, threshes in stats.items():
                low = threshes["Low"]
                high_invert = threshes["High"]["Inverted"]
                high_logistic = threshes["High"]["Logistic"]
                # sample + "Time " + time + " Inverted"
                # sample + "Time " + time + " Logistic"
                orig_rgb = self._grayscale_to_rgb(time_separated_stack[int(t)])
                invert_rgb = np.clip(self._rgb_overlay(time_separated_stack[int(t)], low, high_invert)*255 + orig_rgb, 0, 255)
                logistic_rgb = np.clip(self._rgb_overlay(time_separated_stack[int(t)], low, high_logistic)*255 + orig_rgb, 0, 255)
                if mip:
                    orig_mip = np.amax(time_separated_stack[int(t)], axis=0)
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    fig.suptitle(sample)
                    ax1.imshow(orig_mip)
                    ax1.set_title("Original")
                    ax2.imshow(np.amax(invert_rgb, axis=0))
                    ax2.set_title("Inverted")
                    ax3.imshow(np.amax(logistic_rgb, axis=0))
                    ax3.set_title("Logistic")
                    plt.show()
                else:
                    for z in range(invert_rgb.shape[0]):
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        fig.suptitle(sample + " slice " + str(z))
                        ax2.imshow(invert_rgb[z])
                        ax2.set_title("Inverted")
                        ax3.imshow(logistic_rgb[z])
                        ax3.set_title("Logistic")
                        plt.show()

    def _compare_threshold_differences(self):
        """
        This method determines the distances between the inverted and logistic methods for each option. These distances are stored in a class dictionary
        :return:
        """
        if len(self.sample_thresholds) == 0:
            print("Thresholds are not yet calculated")
        else:
            distance_values = {"Sample": [], "Time": [], "Option": [], "Distance": []}
            for sample, stats in self.sample_thresholds.items():
                image_stack = io.imread(self.image_paths[sample])
                grayscale_stack = self._grayscale(image_stack)
                time_separated_stack = self._timeframe_sep(grayscale_stack, sample)
                for t, threshes in stats.items():
                    low = threshes["Low"]
                    high_invert = threshes["High"]["Inverted"]
                    high_logistic = threshes["High"]["Logistic"]
                    orig_rgb = self._grayscale_to_rgb(time_separated_stack[int(t)])
                    invert_rgb = self._rgb_overlay(time_separated_stack[int(t)], low, high_invert) * orig_rgb
                    logistic_rgb = self._rgb_overlay(time_separated_stack[int(t)], low, high_logistic) * orig_rgb
                    for r in range(3):
                        distance = self._image_distance(invert_rgb[..., r], logistic_rgb[..., r])
                        distance_values["Sample"].append(sample)
                        distance_values["Time"].append(int(t))
                        distance_values["Option"].append(r)
                        distance_values["Distance"].append(distance)
            df = pd.DataFrame.from_dict(distance_values)
            self.distance_metrics = df

    def _grayscale_to_rgb(self, image):
        """
        This method takes a grayscale image and converts it into a gray RGB image
        :param image:
        :return:
        """
        return np.stack((image,)*3, axis=-1)

    def load_threshold_values(self, load_path):
        """
        This method loads the .json file containing the thresholds at the path.
        :param load_path:
        :return:
        """
        with open(load_path, "r") as j:
            results = json.load(j)
            self.sample_thresholds = results
        j.close()

    def _image_distance(self, image1, image2):
        """
        This method will determine the distance between two images. This distance will represent similarity where a smaller distance implies greater similarity.
        :param image1:
        :param image2:
        :return:
        """
        counts1, centers1 = histogram(image1)
        counts2, centers2 = histogram(image2)
        distance = 0
        voxel_values1 = [counts1[i] if i in centers1 else 0 for i in range(256)]
        voxel_values2 = [counts2[i] if i in centers2 else 0 for i in range(256)]
        for i in range(256):
            distance += math.pow((voxel_values1[i] - voxel_values2[i]), 2)
        return math.sqrt(distance)

    def _efficient_hysteresis_iterative(self, image, low):
        """
        In this method the image and low thresholds are supplied to generate the mask_low labels for all elements greater than the low threshold.
        The high threshold is a maximum value and there will be iteration from the low value to the high value to produce a range of mask_high arrays.
        The aim of this efficiency wise is to improve the speed of mask_high generation for all high thresholds (i) greater than low (i=0)>
        Thus mask_high[i=0] = image > high but all subsequent mask_high[i] for i>0 will be based on the True locations in mask_high[i-1].
        This method is based off of apply_hysteresis_threshold in scikit-image
        (https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/filters/thresholding.py#L1159-L1203)
        :param image: ndArray
        :param low: float
        :return:
        """
        img_max = image.max()
        thresh_width = img_max - low
        mask_low = image > low
        labels_low, num_labels = ndi.label(mask_low)
        thresholded = [None]*thresh_width
        intensities = list(range(low+1, img_max+1))
        num_range = np.arange(num_labels + 1)
        mask_prior = True
        for i in range(len(intensities)):
            i_sum, mask_prior = self._check_greater(image, intensities[i], labels_low, num_range, mask_prior)
            thresholded[i] = i_sum

        thresholded.reverse()
        intensities.reverse()

        return intensities, thresholded

    def _check_greater(self, im, thresh, labels_low, num_range, prior=True):
        mask_high = np.greater(im, thresh, where=prior)
        sums = ndi.sum_labels(mask_high, labels_low, num_range)
        connected_to_high = sums > 0
        thresholded = connected_to_high[labels_low].astype('uint8')
        return thresholded.sum(), mask_high

    def test_iter_hyst(self):
        image_details = self.file_list[0]
        print(image_details[1])
        image = io.imread(image_details[0])
        gray_image = self._grayscale(image)
        low_thresh, valid_low = self._low_select(gray_image)
        orig_start_time = time.process_time()
        intens1, voxels1 = self._iterate_hysteresis(gray_image, low_thresh)
        orig_end_time = time.process_time()
        eff_start_time = time.process_time()
        intens2, voxels2 = self._efficient_hysteresis_iterative(gray_image, low_thresh)
        eff_end_time = time.process_time()
        print("Time difference:\n" + str(orig_end_time - orig_start_time) + "s original\n" + str(eff_end_time - eff_start_time) + "s Efficient")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(intens1, voxels1)
        ax2.plot(intens2, voxels2)
        ax1.set_title("Original")
        ax2.set_title("New")
        plt.show()

    def explore_distance_metrics(self):
        """
        This method determines the distances between the invert and logistic versions for each option. These metrics are put into a dataframe and the average
        across the samples is determined for each option.
        :return:
        """
        if len(self.image_paths) == 0:
            self.process_images()
        self._compare_threshold_differences()
        categories = list(self.distance_metrics.columns)
        categories.remove('Sample')
        categories.remove('Time')
        categories.remove('Distance')
        distance_data = self.distance_metrics.groupby(categories)['Distance'].mean()
        print(distance_data)

    def threshold_images(self, save_path, version=[1], excluded_type=["Inverted"]):
        if len(self.sample_thresholds) == 0:
            for f in self.file_list:
                image = io.imread(f[0])
                gray_image = self._grayscale(image)
                image_set = self._timeframe_sep(gray_image, f[1])
                for img in image_set:
                    high_threshes = self._specific_thresholds(img, excluded_type, version)
                    if high_threshes is not None:
                        for thresh_type, thresholds in high_threshes.items():
                            image_mask = self._threshold_image(img, thresholds[0], thresholds[1]).astype("int")
                            thresholed_image = (img*image_mask).astype("uint8")
                            tifffile.imwrite(save_path + f[1].split('.tif')[0] + "ThreshType" + thresh_type + "Option" + str(thresholds[2]) +
                                             ".tif", thresholed_image, imagej=True)
        else:
            self.process_images()
            progress_count = 0
            for sample, stats in self.sample_thresholds.items():
                image_stack = io.imread(self.image_paths[sample])
                grayscale_stack = self._grayscale(image_stack)
                time_separated_stack = self._timeframe_sep(grayscale_stack, sample)
                for t, threshes in stats.items():
                    time_point = int(t)
                    img = time_separated_stack[time_point]
                    low_thresh = float(threshes["Low"])
                    for thresh_types, thresh in threshes["High"].items():
                        for options, option_results in thresh.items():
                            image_mask = self._threshold_image(img, low_thresh, low_thresh+float(option_results)).astype("int")
                            thresholed_image = (image_mask*img).astype("uint8")
                            tifffile.imwrite(save_path + sample.split('.tif')[0] + "-ThreshType#" + thresh_types + "-Option#" + options +
                                             ".tif", thresholed_image, imagej=True)
                progress_count += 1
                if progress_count == 1:
                    print("Finished with 1 sample.")
                else:
                    print("Finished with " + str(progress_count) + " samples.")

    def threshold_permutations(self, specific_files, steepness, power, save_path):
        if type(steepness) is not list:
            steepness = [steepness]
        if type(power) is not list:
            power = [power]
        threshold_recording = {}
        for p in power:
            for k in steepness:
                for f in self.file_list:
                    if f[1] in specific_files:
                        print(f[1])
                        image = io.imread(f[0])
                        gray_image = self._grayscale(image)
                        image_set = self._timeframe_sep(gray_image, f[1])
                        for i in range(image_set.shape[0]):
                            high_threshes = self._specific_thresholds(image_set[i], ["Inverted"], [0, 1, 2], steepness=k, power=p)
                            self._dict_for_json([f[1], "High", (p, k)], high_threshes["Logistic"], threshold_recording)
                            steep_label = str(k) if k/10 >= 1 else "0" + str(k)
                            if high_threshes is not None:
                                for thresh_type, thresholds in high_threshes.items():
                                    for opt, details in thresholds.items():
                                        thresholed_image = self._orig_mask_mip(image_set[i], details[0], details[1])
                                        '''image_mask = self._threshold_image(image_set[i], thresholds[0], thresholds[0] + thresholds[1]).astype("int")
                                        thresholed_image = (image_set[i] * image_mask).astype("uint8")'''
                                        io.imsave(save_path + f[1].split('.tif')[0] + "-Steep#" + steep_label + "-Power#" + str(p) + "-Option#" +
                                                         str(opt) + ".tif", thresholed_image)
                self.sample_thresholds = {}
        print(threshold_recording)
        with open(save_path + "TestResults.json", 'w') as j:
            json.dump(threshold_recording, j)

    def _dict_for_json(self, key, value, dictionary):
        val_holder = self._list_to_string(value)
        print("Recorded Value", val_holder)
        self._dict_key_extend(key, val_holder, dictionary)

    def _list_to_string(self, structure):
        if type(structure) is list:
            results = []
            for l in structure:
                if type(l) is list or type(l) is dict:
                    result = self._list_to_string(l)
                else:
                    result = str(l)
                results.append(result)
            return results
        elif type(structure) is dict:
            results = {}
            for k, v in structure.items():
                if type(v) is list or type(v) is dict:
                    results[k] = self._list_to_string(v)
                else:
                    results[k] = str(v)
            return results
        else:
            return str(structure)


    def _dict_key_extend(self, key, value, dictionary):
        """
        This method will extend a dictionary when needed or step down a dictionary to add a value.
        :param key: This will be a list of the keys. The order defines the order of keys in the nested dict
        :param value: This will be assigned to the last key in the nested dictionaries
        :param dictionary: This is the source dictionary to be referenced
        :return:
        """
        key0 = key[0]
        if type(key0) is tuple:
            key0 = ""
            for i in range(len(key[0])):
                key0 += str(key[0][i])
                if i != len(key[0]):
                    key0 += " "
        if len(key) > 1:
            if key0 not in dictionary:
                dictionary[key0] = {}
            self._dict_key_extend(key[1:], value, dictionary[key0])

        else:
            dictionary[key0] = value
            return


if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    threshold_comparer = AutoThresholder(input_path)
    # threshold_comparer.load_threshold_values("C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\CompareResults.json")
    # threshold_comparer.process_images()
    # threshold_comparer.get_threshold_results()
    # threshold_comparer.save_threshold_results("C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\CompareResults.json")
    # threshold_comparer.get_threshold_results()
    # threshold_comparer.preview_thresholds_options(True)
    # threshold_comparer.explore_distance_metrics()
    # threshold_comparer.threshold_images("C:\\RESEARCH\\Mitophagy_data\\Time_split\\Thresholded\\", version=[0, 1, 2], excluded_type=[])
    # "N2CCCP+Baf_3C=0T=0.tif", "N2Con_3C=0T=0.tif"
    threshold_comparer.threshold_permutations(["CCCP_1C=0T=0.tif", "N2CCCP+Baf_3C=0T=0.tif", "N2Con_3C=0T=0.tif"], [6, 50], [1],
                                              "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Test_Threshold\\")
    # threshold_comparer.test_iter_hyst()