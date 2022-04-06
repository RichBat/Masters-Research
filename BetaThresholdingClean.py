import os.path
from os import listdir
from os.path import isfile, join, exists

import time
import pandas as pd
import math
from skimage import data, io
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, gaussian, threshold_li, threshold_yen, threshold_triangle, threshold_minimum, threshold_mean
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import cv2
import numpy as np
from knee_locator import KneeLocator
import sample_checker
import json

#This file is similar to AutoHysteresis.py but is cleaner with less clutter and contains only currently relevant functionality

manual_Hysteresis = {"CCCP_1C=0.tif": [[0.1, 0.408], [0.1, 0.25]], "CCCP_1C=1.tif": [[0.116, 0.373], [0.09, 0.22]],"CCCP_2C=0.tif": [[0.107, 0.293], [0.09, 0.2]], "CCCP_2C=1.tif": [[0.09, 0.372], [0.08, 0.15]],"CCCP+Baf_2C=0.tif": [[0.093, 0.279], [0.1, 0.17]], "CCCP+Baf_2C=1.tif": [[0.098, 0.39], [0.1, 0.35]],"Con_1C=0.tif": [[0.197, 0.559], [0.14, 0.18]], "Con_1C=2.tif": [[0.168, 0.308], [0.11, 0.2]],"Con_2C=0.tif": [[0.219, 0.566], [0.19, 0.31]], "Con_2C=2.tif": [[0.137, 0.363], [0.13, 0.23]],"HML+C+B_2C=0.tif": [[0.102, 0.55], [0.14, 0.31]], "HML+C+B_2C=1.tif": [[0.09, 0.253], [0.09, 0.18]],"HML+C+B_2C=2.tif": [[0.114, 0.477], [0.11, 0.31]], "LML+C+B_1C=0.tif": [[0.09, 0.152], [0.05, 0.1]],"LML+C+B_1C=1.tif": [[0.102, 0.232], [0.07, 0.15]], "LML+C+B_1C=2.tif": [[0.034, 0.097], [0.024, 0.1]]}

def testing():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Output\\"
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results_record = ""
    extra_noise = False
    complete_json_results = {}
    calculate_original = False
    number_of_samples = len(images)
    sample_progress = 0
    actual_beginning_time = time.process_time()
    for filename in images:
        sample_progress += 1
        json_sample_results = {
            "Total Voxels": None,
            "Filter_Cutoff": None,
            "All Filter Thresh": None,
            "Sample Variants": []
        }
        img = io.imread(input_path + filename)
        #print("Hist of original image:", filename)
        #show_hist(img)
        noise_adjustment = sample_checker.ThresholdOutlierDetector(input_path, filename)
        noise_adjustment_p = sample_checker.ThresholdOutlierDetector(input_path, filename, 'poisson')
        results_record = results_record + "Sample=" + filename + "\n"
        if extra_noise:
            for noise_ratio in range(60, 101, 20):
                print("Noise percentage", noise_ratio)
                noise_variant_time = time.process_time()
                noisy_image = noise_adjustment.generate_noisy_image(noise_ratio / 100)
                noisy_image_p = noise_adjustment_p.generate_noisy_image(noise_ratio / 100)
                low_thresh, glow_thresh = testing_knee(noisy_image, cutoff=1, log_hist=True)
                print("Poisson Noise Test:")
                print("Poisson Noise Low Thresh", testing_knee(noisy_image_p, cutoff=1, log_hist=True))
                print("---------------------------------------")
                print("Sample:", filename, "Low threshold:", low_thresh)
                #show_hist(noisy_image, low_thresh, 0.25)
                loop_time = time.process_time()
                high_thresh = high_threshold_nested(noisy_image, low_thresh, 0.25, 0.1)
                print("Time for high thresh:", time.process_time() - loop_time)
                thresholded_image = apply_hysteresis_threshold(noisy_image, low_thresh, high_thresh)
                thr_voxels = int(np.round(thresholded_image/np.max(thresholded_image)).sum())
                print("Thresholded Voxels outside of Outlier Detection:")
                otsu_cutoff = 0.001
                sufficient, threshold_voxels, otsu_voxels = noise_adjustment.outlierDetection(thresholded_image, otsu_cutoff)
                print("Time for noise variation:", time.process_time() - noise_variant_time)
                results_record = results_record + "For a noise ratio of " + str(noise_ratio / 100) + " a low threshold of " + str(
                    low_thresh) + " and a high threshold of " + str(high_thresh) + " is determined. Number of Hysteresis voxels = " + str(
                    threshold_voxels) + " Number of Otsu Voxels*" + str(otsu_cutoff) + ": " + str(
                    otsu_voxels) + " Were there sufficient voxels? " + str(sufficient) + "\n"
        else:
            #print("Calculating Thresholds")
            filter_results = other_threshold_results(img)
            json_sample_results["All Filter Thresh"] = filter_results
            #print("Other threshold values added", json_sample_results)
            otsu_cutoff = 0.0008
            json_sample_results = addResultsToDictionary(json_sample_results, "Filter_Cutoff", float(otsu_cutoff))
            total_voxels = 0
            filename = filename.split('Noise')[0] + '.tif'
            manual_parameters = manual_Hysteresis[filename]
            progress_count = 0
            density_config = [1, 22, 5]
            prec_config = [2, 15, 2]
            filter_variant_count = len(list(filter_results))
            total_steps = int(((density_config[1]-density_config[0])/density_config[2]) + 1)*int(((prec_config[1]-prec_config[0])/prec_config[2]) + 1)*filter_variant_count
            print("Total steps", total_steps)
            for filt, low_filt in filter_results.items():
                for starting_density in range(density_config[0], density_config[1], density_config[2]):
                    for prec in range(prec_config[0], prec_config[1], prec_config[2]):
                        variant_time_start = time.process_time()
                        progress_count += 1
                        sample_variation_results = {
                            "Precision": None,
                            "Starting Density": None,
                            "Filter Type": None,
                            "Filter_Thresh": None,
                            "Low_Thresh": None,
                            "Orig_Low": None,
                            "High_Thresh": None,
                            "Orig_High": None,
                            "Hyst_Voxels": None,
                            "Filter_Voxels": None,
                            "Sufficient": None,
                            "Intensity Range": None,
                            "Voxels per intensity": None,
                            "Voxels changes": [],
                            "Richard Error": None,
                            "Rensu Error": None
                        }
                        low_thresh, original_low, otsu_thresh = testing_knee(img, cutoff=1, filter_value=low_filt, log_hist=True)
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Filter_Thresh", float(otsu_thresh))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Low_Thresh", float(low_thresh))
                        if calculate_original:
                            sample_variation_results = addResultsToDictionary(sample_variation_results, "Orig_Low", float(original_low))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Starting Density", starting_density)
                        precision = prec/100
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Precision", precision)
                        high_thresh, voxels_per_intensity, intensities, total_voxels = high_threshold_nested(img, low_thresh, starting_density/100, precision)
                        #voxel_changes = change_between_intensities(voxels_per_intensity)
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "High_Thresh", float(high_thresh))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Voxels per intensity", voxels_per_intensity)
                        #sample_variation_results = addResultsToDictionary(sample_variation_results, "Voxels changes", voxel_changes)
                        #sample_variation_results = addResultsToDictionary(sample_variation_results, "Intensity Range", intensities)
                        if calculate_original:
                            original_high, voxels_per_intensity, intensities, total_voxels = high_threshold_nested(img, original_low, starting_density/100, precision)
                            sample_variation_results = addResultsToDictionary(sample_variation_results, "Orig_High", float(original_high))
                            orig_thresholded_image = apply_hysteresis_threshold(img, original_low, original_high)
                            orig_rich, orig_rensu = image_MAE(img, manual_parameters, orig_thresholded_image)
                            sample_variation_results = addResultsToDictionary(sample_variation_results, "Richard Error", float(orig_rich))
                            sample_variation_results = addResultsToDictionary(sample_variation_results, "Rensu Error", float(orig_rensu))
                        thresholded_image = apply_hysteresis_threshold(img, low_thresh, high_thresh)
                        rich_error, rensu_error = image_MAE(img, manual_parameters, thresholded_image)
                        sufficient, threshold_voxels, otsu_voxels = noise_adjustment.outlierDetection(thresholded_image, otsu_cutoff, True)
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Richard Error", float(rich_error))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Rensu Error", float(rensu_error))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Hyst_Voxels", float(threshold_voxels))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Filter_Voxels", float(otsu_voxels))
                        sample_variation_results = addResultsToDictionary(sample_variation_results, "Sufficient", int(sufficient))
                        results_record = results_record + "Precision: " + str(precision) + "\t A low threshold of " + str(
                            low_thresh) + " and a high threshold of " + str(
                            high_thresh) + " are determined. Number of Hysteresis voxels = " + str(
                            threshold_voxels) + " Number of Otsu Voxels*" + str(otsu_cutoff*100) + ": " + str(
                            otsu_voxels) + " Were there sufficient voxels? " + str(
                            sufficient) + "\n"
                        #json_results[filename] = addResultsToDictionary()
                        if starting_density % 11 == 0 and prec % 4 == 0:
                            if not os.path.exists(output_path + filt + "\\"):
                                os.makedirs(output_path + filt + "\\")
                            io.imsave(output_path + filt + "\\" + "Density" + str(starting_density) + "Precision" + str(precision) + filename, thresholded_image)
                        json_sample_results["Sample Variants"].append(sample_variation_results)
                        print("Progress: " + filename + " is " + str(int((progress_count/total_steps)*100)) + "%")
                        print("Time taken for this variant: " + str(time.process_time()-variant_time_start))
                        print("Total time thus far: " + str(time.process_time() - actual_beginning_time))
                        print("Sample number " + str(sample_progress) + " of " + str(number_of_samples))
            json_sample_results = addResultsToDictionary(json_sample_results, "Total Voxels", int(total_voxels))
            with open(output_path + filename + 'Results.json', 'w') as j:
                json.dump(json_sample_results, j)
            complete_json_results[filename] = json_sample_results
    #print(results_record)
    '''f = open(output_path + "ResultsOfThresholding.txt", "w")
    f.write(results_record)
    f.close()'''
    print(complete_json_results)
    with open(output_path + 'CompleteResults.json', 'w') as j:
        json.dump(complete_json_results, j)


def change_between_intensities(intensities):
    list_of_intensities = list(intensities)
    results = []
    for k in range(1, len(list_of_intensities), 1):
        perc = float((intensities[list_of_intensities[k]]/intensities[list_of_intensities[k-1]]) - 1)
        results.append(perc)
    return results


def save_hist(img, low, population, save_path):
    counts, centers = histogram(img)
    low_index = np.where(centers == low)[0][0]
    counts = counts[low_index:]
    centers = centers[low_index:]
    total_population = counts.sum()
    initial_density = int(total_population*population)
    starting_intensity = 0
    for intensity in range(len(counts), 0, -1):
        if np.sum(counts[intensity:]) >= initial_density:
            starting_intensity = centers[intensity]
            break
    plt.plot(centers, counts, color='black')
    plt.axvline(low, 0, 1, label='Low', color="red")
    plt.axvline(starting_intensity, 0, 1, label='High', color="blue")
    plt.savefig(save_path)
    plt.clf()


def show_hist(img, low=None, population=None):
    counts, centers = histogram(img)
    plt.plot(centers, counts, color='black')
    print("Voxels:", counts.sum())
    if low != None:
        low_index = np.where(centers == low)[0][0]
        print("Low values", low, low_index, centers[low_index])
        counts = counts[low_index:]
        centers = centers[low_index:]
        plt.axvline(low, 0, 1, label='Low', color="red")

    total_population = counts.sum()
    print("Current Voxels:", total_population)
    if population != None:
        initial_density = int(total_population*population)
        starting_intensity = 0
        for intensity in range(len(counts), 0, -1):
            if np.sum(counts[intensity:]) >= initial_density:
                starting_intensity = centers[intensity]
                break
        print("Starting Intensity:", starting_intensity)
        starting_index = np.where(starting_intensity == centers)[0][0]
        print(starting_index)
        print("Starting_voxels", counts[starting_index:].sum())
        plt.axvline(starting_intensity, 0, 1, label='High', color="blue")

    plt.show()
    plt.clf()


def high_threshold_nested(img, low, start_density, decay_rate=0.1):
    voxels_by_intensity, intensities = histogram(img, nbins=256) #This acquires the voxels at each intensity and the intensity at each element
    #print("Intensities:", intensities, " Low: ", low)
    low_index = np.where(intensities == low)[0][0]
    voxels_by_intensity = voxels_by_intensity[low_index:]
    intensities = intensities[low_index:]
    img = np.pad(img, 1)
    voxels_per_intensity = []
    #Now the majority of low intensity and noise voxels are discarded
    template_compare_array = np.zeros_like(img) + 1  # Array of 1's which can be multiplied by each threshold
    total_population = voxels_by_intensity.sum()
    initial_density = int(total_population*start_density) #This shall get the number of voxels above the initial high threshold
    starting_intensity = intensities[-1]
    for intensity in range(len(voxels_by_intensity), 0, -1):
        if np.sum(voxels_by_intensity[intensity:]) >= initial_density:
            starting_intensity = intensities[intensity]
            #print("Starting intensity found", starting_intensity)
            break

    #Now the starting high threshold has been acquired the rest can begin
    range_of_intensities = recursive_intensity_steps(low, starting_intensity, decay_rate) #This will provide a list of intensities with which to check for neighbours
    range_of_intensities.insert(0, int(starting_intensity))
    array_of_structures = np.zeros_like(img)
    #print("Range of intensities:", range_of_intensities)
    range_of_intensities_time_start = time.process_time()
    for i in range_of_intensities:
        compare_array = template_compare_array * i
        results = np.greater_equal(img, compare_array).astype(int)
        array_of_structures = array_of_structures + results
    range_of_intensities_time_end = time.process_time()
    #print("Range of intensities time:", range_of_intensities_time_end-range_of_intensities_time_start)
    viable_values = np.argwhere(array_of_structures > 0)
    number_of_viable_structures = viable_values.shape[0]
    #print("Number of viable voxels:", number_of_viable_structures)
    '''if number_of_viable_structures > 1000000:
        print("TOO MANY STRUCTURES")
        return None'''
    starting_position = np.max(array_of_structures) #second highest value for highest intensity voxels will be the first set of neighbours
    adjacency_array = np.zeros_like(array_of_structures) #array of zeros. Will be filled with 1's for joins
    adjacency_array[np.where(array_of_structures == starting_position)] = 1 #Initial highest structures for join array
    old_voxels = np.sum(adjacency_array)
    voxels_per_intensity.append(int(old_voxels))
    initial_voxels = old_voxels
    change_by_intensity = []
    progress = "With initially " + str(old_voxels) + " voxels for core structures.\n"
    voxel_intensities = range_of_intensities
    for r in range(1, len(range_of_intensities), 1):
        for v in viable_values:
            if array_of_structures[v[0], v[1], v[2]] == r and np.any(adjacency_array[v[0]-1:v[0]+2, v[1]-1:v[1]+2, v[2]-1:v[2]+2]):
                adjacency_array[v[0], v[1], v[2]] = 1
        for rv in np.flip(viable_values, axis=0):
            if array_of_structures[rv[0], rv[1], rv[2]] == r and np.any(adjacency_array[rv[0]-1:rv[0]+2, rv[1]-1:rv[1]+2, rv[2]-1:rv[2]+2]):
                adjacency_array[rv[0], rv[1], rv[2]] = 1
        new_voxels = np.sum(adjacency_array)
        voxels_per_intensity.append(int(new_voxels))
        change = (new_voxels/old_voxels) - 1
        progress = progress + "For an intensity of " + str(range_of_intensities[r]) + " the change in voxels is " + str(
            change) + " and the total change is " + str((new_voxels - initial_voxels) / initial_voxels) + "\n"
        old_voxels = new_voxels
        change_by_intensity.append(change)
    average_of_change = sum(change_by_intensity) / (len(range_of_intensities) - 1)
    #print("The average of the change in voxels is", average_of_change)
    answer = 0
    for cbi in range(0, len(change_by_intensity), 1):
        if change_by_intensity[cbi] <= average_of_change:
            #print("The best threshold is at", range_of_intensities[cbi+1])
            answer = range_of_intensities[cbi+1]
            break
    #print(progress)
    return answer, voxels_per_intensity, voxel_intensities, total_population


def calculate_high_threshold(img, low, start_density, decay_rate=0.1, faster=True):
    '''
    - low is used to remove the low intensity and noise voxels which will greatly skew the density
    - start_density to acquire the high intensity threshold starting point. Voxels above this intensity are within this ratio
    - stop_margin is the margin of voxels gained to determine a stopping point. This should be a ratio such that voxels_retained/reference
    where reference will either be the prior voxels retained or the total voxels retained. The ratio needs to be decided
    - decay_rate is the rate at which the threshold value reduces by for each neighbouring check
    '''
    #print("Calculation beginning", low)
    voxels_by_intensity, intensities = histogram(img, nbins=256) #This acquires the voxels at each intensity and the intensity at each element
    low_index = np.where(intensities == low)[0][0]
    voxels_by_intensity = voxels_by_intensity[low_index:]
    intensities = intensities[low_index:]
    img = np.pad(img, 1)
    #Now the majority of low intensity and noise voxels are discarded
    template_compare_array = np.zeros_like(img) + 1  # Array of 1's which can be multiplied by each threshold
    total_population = voxels_by_intensity.sum()
    initial_density = int(total_population*start_density) #This shall get the number of voxels above the initial high threshold
    starting_intensity = intensities[-1]
    for intensity in range(len(voxels_by_intensity), 0, -1):
        if np.sum(voxels_by_intensity[intensity:]) >= initial_density:
            starting_intensity = intensities[intensity]
            #print("Starting intensity found", starting_intensity)
            break

    #Now the starting high threshold has been acquired the rest can begin
    range_of_intensities = recursive_intensity_steps(low, starting_intensity, decay_rate) #This will provide a list of intensities with which to check for neighbours
    range_of_intensities.insert(0, starting_intensity)
    array_of_structures = np.zeros_like(img)

    range_of_intensities_time_start = time.process_time()
    for i in range_of_intensities:
        compare_array = template_compare_array * i
        results = np.greater_equal(img, compare_array).astype(int)
        array_of_structures = array_of_structures + results
    range_of_intensities_time_end = time.process_time()
    #print("Time taken for intensity discretisation:", range_of_intensities_time_end-range_of_intensities_time_start)

    #array_of_structures an array of ints. The voxel values are proportional to the number of intensities from range_of_intensities that the voxel is greater than/equal to
    #print("Range of intensities:", range_of_intensities)
    starting_position = np.max(array_of_structures) #second highest value for highest intensity voxels will be the first set of neighbours
    adjacency_array = np.zeros_like(array_of_structures) #array of zeros. Will be filled with 1's for joins
    adjacency_array[np.where(array_of_structures == starting_position)] = 1 #Initial highest structures for join array
    old_voxels = np.sum(adjacency_array)
    initial_voxels = old_voxels
    progress = "With initially " + str(old_voxels) + " voxels for core structures.\n"
    change_by_intensity = []
    nditer_time_total = 0
    inefficient_time_total = 0
    time_total = 0
    for r in range(1, len(range_of_intensities), 1):
        #print("At intensity:", range_of_intensities[r])
        counter = 0
        repeat = True
        repeat_prior_voxels = old_voxels
        neighbour_validation_time_start = time.process_time()
        time_per_repeat = []
        '''while repeat:
            repeat_time_start = time.process_time()
            for x in range(1, array_of_structures.shape[0]-1, 1):
                for y in range(1, array_of_structures.shape[1] - 1, 1):
                    for z in range(1, array_of_structures.shape[2] - 1, 1):
                        if array_of_structures[x, y, z] == r and np.any(adjacency_array[x-1:x+2, y-1:y+2, z-1:z+2]):
                            adjacency_array[x, y, z] = 1
            repeat_new_voxels = np.sum(adjacency_array)
            iterative_changes = (repeat_new_voxels-repeat_prior_voxels)/repeat_prior_voxels
            ''''''if counter > 0:
                print("Increase in voxels by", iterative_changes*100, " for repeat number", counter-1, " with", int(iterative_changes*repeat_prior_voxels), " new voxels")
                print("New voxels", repeat_new_voxels, " prior voxels", repeat_prior_voxels)''''''
            if counter >= 9 or repeat_new_voxels == repeat_prior_voxels:
                #print("Number of loops", counter)
                repeat = False
            repeat_prior_voxels = repeat_new_voxels
            counter += 1
            repeat_time_end = time.process_time()
            time_per_repeat.append(repeat_time_start-repeat_time_end)'''
        if faster:
            nditer_time_start = time.process_time()
            it = np.nditer(array_of_structures, flags=['multi_index'])
            reverse_it = np.nditer(np.flip(array_of_structures), flags=['multi_index'])
            for ele in it:
                z = it.multi_index[0]
                y = it.multi_index[1]
                x = it.multi_index[2]
                if ele == r and np.any(adjacency_array[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]):
                    adjacency_array[z, y, x] = 1
            for r_ele in reverse_it:
                z = reverse_it.multi_index[0]
                y = reverse_it.multi_index[1]
                x = reverse_it.multi_index[2]
                if r_ele == r and np.any(adjacency_array[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]):
                    adjacency_array[z, y, x] = 1
            time_total += time.process_time() - nditer_time_start
        else:
            inefficient_time_start = time.process_time()
            for x in range(1, array_of_structures.shape[0]-1, 1):
                for y in range(1, array_of_structures.shape[1] - 1, 1):
                    for z in range(1, array_of_structures.shape[2] - 1, 1):
                        if array_of_structures[x, y, z] == r and np.any(adjacency_array[x-1:x+2, y-1:y+2, z-1:z+2]):
                            adjacency_array[x, y, z] = 1
            for x in range(array_of_structures.shape[0] - 1, 1, -1):
                for y in range(array_of_structures.shape[1] - 1, 1, -1):
                    for z in range(array_of_structures.shape[2] - 1, 1, -1):
                        if array_of_structures[x, y, z] == r and np.any(
                                adjacency_array[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]):
                            adjacency_array[x, y, z] = 1
            time_total += time.process_time() - inefficient_time_start
        #The double for loop above should capture sufficient voxels compared to the while loop as that would repeat loops to the right to build structures to the left
        new_voxels = np.sum(adjacency_array)
        change = (new_voxels/old_voxels) - 1
        progress = progress + "For an intensity of " + str(range_of_intensities[r]) + " the change in voxels is " + str(change) + " and the total change is " + str((new_voxels - initial_voxels)/initial_voxels) + "\n"
        old_voxels = new_voxels
        change_by_intensity.append(change)
        neighbour_validation_time_end = time.process_time()
        #print("Time taken for intensity", range_of_intensities[r], " took", neighbour_validation_time_end-neighbour_validation_time_start)
    average_of_change = sum(change_by_intensity)/(len(range_of_intensities) - 1)
    #print("The average of the change in voxels is", average_of_change)
    answer = 0
    for cbi in range(0, len(change_by_intensity), 1):
        if change_by_intensity[cbi] <= average_of_change:
            #print("The best threshold is at", range_of_intensities[cbi+1])
            answer = range_of_intensities[cbi+1]
            break
    #print(progress)
    return answer


def recursive_intensity_steps(bottom, top, ratio):
    new_intensity = int(top * (1 - ratio))  # this will reduce top by 10%
    if new_intensity == top:
        new_intensity = top - 1  # Will avoid duplicate percentiles if the percentage cause a reduction of less than 1
    steps = [new_intensity]
    if int(new_intensity * (1 - ratio)) > bottom:
        results = recursive_intensity_steps(bottom, new_intensity, ratio)
        for r in results:
            steps.append(int(r))
    return steps


def testing_knee(img, cutoff=1, filter_value=None, log_hist=False):
    #print("Histogram for knee")
    counts, centers = histogram(img, nbins=256)

    gaussian_image = gaussian(img)
    rescaled_gaussian_image = (gaussian_image/np.max(gaussian_image))*np.max(img)
    #print("Rescaled Gaussian Otsu", threshold_otsu(rescaled_gaussian_image))
    norm_gaussian = (gaussian_image/np.max(gaussian_image))*np.max(img)
    gaussian_counts, gaussian_centers = histogram(norm_gaussian, nbins=256)
    gaussian_counts, gaussian_centers = ((gaussian_counts/np.max(gaussian_counts))*np.max(counts)).astype('int'), ((gaussian_centers / np.max(gaussian_centers)) * np.max(centers)).astype('int')

    if cutoff < centers[0]:
        cut = 1
    else:
        cut = np.where(centers == cutoff)[0][0]
    counts = counts[cut:]
    centers = centers[cut:]
    #print("Final Counts: ", counts[-20:-1])
    if log_hist:
        counts = np.where(counts != 0, np.log10(counts), 0)
        gaussian_counts = np.where(gaussian_counts != 0, np.log10(gaussian_counts), 0)
    #print(centers.shape)
    '''
    plt.figure(figsize=(6, 6))
    plt.plot(centers, counts, color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("Histogram")
    plt.tight_layout()
    plt.show()'''

    safe_knee = True
    if filter_value is None:
        otsu_thresh = threshold_otsu(img)
    else:
        otsu_thresh = filter_value
    #print("Otsu Thresh:", otsu_thresh)
    #print("Otsu of Guassian", threshold_otsu(norm_gaussian))
    gaussian_otsu = threshold_otsu(norm_gaussian)
    true_knee = 0
    knee_found = True
    gaussian_knee = 0
    first_knee = int(KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing").knee)
    while safe_knee:
        locator = KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing")
        glocator = KneeLocator(x=gaussian_centers, y=gaussian_counts, curve="convex", direction="decreasing")
        knee = int(locator.knee)
        gaussian_knee = int(glocator.knee)
        if knee > otsu_thresh and knee_found:
            true_knee = knee
            knee_found = False
            #print("True Knee", true_knee)
        if knee <= otsu_thresh or gaussian_knee < gaussian_otsu:
            #print("Determined knee", knee, gaussian_knee)
            #print("Standard Intensity", centers[0])
            #print("Gaussian Intensity", gaussian_centers[0])
            #print("Otsu Thresh", otsu_thresh)
            centers = centers[1:]
            counts = counts[1:]
            gaussian_centers = gaussian_centers[1:]
            gaussian_counts = gaussian_counts[1:]
        else:
            safe_knee = False
            #print("Final Standard Intensity", centers[0])
            #print("Final Gaussian Intensity", gaussian_centers[0])
            #print("Determined knee", knee, gaussian_knee)
    if not knee_found:
        first_knee = locator.knee
    gaussian_knee = glocator.knee
    #print("knees: ", locator.all_knees, glocator.all_knees)


    #locator.plot_knee()
    #plt.show()
    #knee = int(locator.norm_knee*255)

    return true_knee, first_knee, otsu_thresh

def hysteresis_thresholding_stack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def addResultsToDictionary(dict_to_add_to, result_key, result_value):
    #Number of parameters and results must match
    if dict_to_add_to[result_key] is not None:
        if type(dict_to_add_to[result_key]) is list:
            dict_to_add_to[result_key].append(result_value)
        else:
            temp = dict_to_add_to[result_key]
            dict_to_add_to[result_key] = [temp, result_value]
    else:
        dict_to_add_to[result_key] = result_value
    return dict_to_add_to

def other_threshold_results(img):
    result = {}
    #threshold_yen, threshold_triangle, threshold_minimum, threshold_mean
    try:
        result["Otsu"] = float(threshold_otsu(img))
    except Exception as e:
        print("Otsu Failed")
        print(e)
    try:
        result["Li"] = float(threshold_li(img))
    except Exception as e:
        print("Li Failed")
        print(e)
    try:
        result["Yen"] = float(threshold_yen(img))
    except Exception as e:
        print("Yen Failed")
        print(e)
    try:
        result["Triangle"] = float(threshold_triangle(img))
    except Exception as e:
        print("Triangle Failed")
        print(e)
    '''try:
        result["Min"] = float(threshold_minimum(img))
    except Exception as e:
        print("Min Failed")
        print(e)'''
    try:
        result["Mean"] = float(threshold_mean(img))
    except Exception as e:
        print("Mean Failed")
        print(e)
    return result

def preview_hysteresis_high(file_name):
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    img = io.imread(input_path + file_name)
    low, __, ___ = testing_knee(img, log_hist=True)
    #show_hist(img, low, 0.25)
    total_array = np.zeros_like(img)
    print("Building Preview")
    voxels_by_intensity, intensities = histogram(img)
    low_index = np.where(intensities == low)[0][0]
    voxels_by_intensity = voxels_by_intensity[low_index:]
    starting_intensity = intensities[-1]
    template_compare_array = np.zeros_like(img) + 1  # Array of 1's which can be multiplied by each threshold
    total_population = voxels_by_intensity.sum()
    initial_density = int(total_population * 0.25)
    for intensity in range(len(voxels_by_intensity), 0, -1):
        if np.sum(voxels_by_intensity[intensity:]) >= initial_density:
            starting_intensity = intensities[intensity]
            print("Starting intensity found", starting_intensity)
            break
    range_of_intensities = recursive_intensity_steps(low, starting_intensity, 0.1) #This will provide a list of intensities with which to check for neighbours
    range_of_intensities.insert(0, starting_intensity)
    array_of_structures = np.zeros_like(img)
    print("Range of intensities:", range_of_intensities)
    for i in range_of_intensities:
        compare_array = template_compare_array * i
        results = np.greater_equal(img, compare_array).astype(int)
        array_of_structures = array_of_structures + results
    slice = int(array_of_structures.shape[0]/2)
    io.imshow(array_of_structures[slice])
    plt.show()
    return total_array

def image_MAE(image, manual_parameters, reference_image):
    first_image = apply_hysteresis_threshold(image, manual_parameters[0][0]*255, manual_parameters[0][1]*255) #Image from my (Richard) parameters
    second_image = apply_hysteresis_threshold(image, manual_parameters[1][0]*255, manual_parameters[1][1]*255) #Image from Rensu's parameters
    automatic_threshold = reference_image #This is not a saved image but is rather in an array
    automatic_threshold = automatic_threshold/np.max(automatic_threshold)
    first_image_error = np.average(np.abs(first_image - automatic_threshold))
    second_image_error = np.average(np.abs(second_image - automatic_threshold))
    return first_image_error, second_image_error

if __name__ == "__main__":
    testing()
    #preview_hysteresis_high('CCCP_1C=0Noise000.tif')