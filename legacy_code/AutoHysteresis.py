from os import listdir
from os.path import isfile, join, exists

import time
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from skimage import data, io
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu, threshold_otsu
from skimage.exposure import histogram, equalize_hist, equalize_adapthist, rescale_intensity
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from knee_locator import KneeLocator
from matplotlib.widgets import Slider
from sklearn.metrics import mean_absolute_error
import sample_checker

manual_Hysteresis = {"CCCP_1C=0.tif": [[0.1, 0.408], [0.1, 0.25]], "CCCP_1C=1.tif": [[0.116, 0.373], [0.09, 0.22]],"CCCP_2C=0.tif": [[0.107, 0.293], [0.09, 0.2]], "CCCP_2C=1.tif": [[0.09, 0.372], [0.08, 0.15]],"CCCP+Baf_2C=0.tif": [[0.093, 0.279], [0.1, 0.17]], "CCCP+Baf_2C=1.tif": [[0.098, 0.39], [0.1, 0.35]],"Con_1C=0.tif": [[0.197, 0.559], [0.14, 0.18]], "Con_1C=2.tif": [[0.168, 0.308], [0.11, 0.2]],"Con_2C=0.tif": [[0.219, 0.566], [0.19, 0.31]], "Con_2C=2.tif": [[0.137, 0.363], [0.13, 0.23]],"HML+C+B_2C=0.tif": [[0.102, 0.55], [0.14, 0.31]], "HML+C+B_2C=1.tif": [[0.09, 0.253], [0.09, 0.18]],"HML+C+B_2C=2.tif": [[0.114, 0.477], [0.11, 0.31]], "LML+C+B_1C=0.tif": [[0.09, 0.152], [0.05, 0.1]],"LML+C+B_1C=1.tif": [[0.102, 0.232], [0.07, 0.15]], "LML+C+B_1C=2.tif": [[0.034, 0.097], [0.024, 0.1]]}

def test_saved_images(substring = None):
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    reference_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    reference_images = [f for f in listdir(reference_path) if isfile(join(reference_path, f))]
    if substring != None:
        reference_imgs = [ri for ri in reference_images if substring in ri]
    else:
        reference_imgs = reference_images
    record = "Errors determined between manual images and automatic thresholding for:\n"
    for src in images:
        sample_name = src.split('.')[0]
        references = [r for r in reference_imgs if sample_name in r]
        src_img = io.imread(input_path + src)
        record = record + "For sample " + src + "\n"
        if len(references) > 0:
            for ref in references:
                ref_img = io.imread(reference_path + ref)
                first_error, second_error = image_MAE(image=src_img, manual_parameters=manual_Hysteresis[src], reference_image=ref_img, reference_index=1)
                record = record + ref + ": Richard Image Error = " + str(first_error) + " | Rensu Image Error = " + str(second_error) + "\n"
    f = open(reference_path + "ImageMAE.txt", "w")
    f.write(record)
    f.close()


def testing():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = ""
    complete_results = {}
    thresholds_per_sample = {}
    output_of_results = ""
    noisy_image_results = ""
    failed_variants = ""
    total_time_per_option = [0, 0]
    print(images)
    for filename in images:
        output_of_results += "Sample: " + filename
        img = io.imread(input_path + filename)
        #print("Otsu: ", threshold_multiotsu(img))
        kernel_size = np.array([img.shape[dim] // 8 for dim in range(img.ndim)])
        kernel_size[0] = img.shape[0]
        #position = testing_knee(img)
        #img = equalize_adapthist(img, kernel_size)
        #img = equalize_hist(img)
        #img = rescale_intensity(img, in_range=(p2, p98))
        #hist_average(img, 5)
        print("Sample", filename)
        position = testing_knee(img, log_hist=True)
        #low1, high = determine_hysteresis_thresholds(img, moving_average_frame=20, cut_off_slope=3, log_value=False)
        #low2, high = determine_hysteresis_thresholds(img, moving_average_frame=30, cut_off_slope=3, log_value=False)
        #low3, high = determine_hysteresis_thresholds(img, moving_average_frame=20, cut_off_slope=4, log_value=False)
        #print("Kneedle low:", position, " Determined low:", low1*255)
        disqualified_list = ""
        if(manual_Hysteresis[filename][0][1] < position or manual_Hysteresis[1][1] < position):
            disqualified_list += filename + "\n"
        #density_evaluate(position, img)
        #output_of_results = output_of_results + " with a low threshold of " + str(position) + "and a determined low of" + str(low1*255) + " " + str(low2*255) + " " + str(low3*255) + "\n"
        #log_position = testing_knee(img)
        high_thresholds = []
        full_run = True
        noisy_image_results = noisy_image_results + "Sample: " + filename + "\n"
        if full_run:
            #hysteresis_high_preview = preview_hysteresis_high(img, position)
            #io.imsave(output_path + '.'.join(filename.split(sep=".")[:-1]) + "Preview.tif", hysteresis_high_preview)
            for density in range(25, 26, 5):
                for precision in range(15, 16, 5):
                    resolution = precision/100
                    end_voxels = density/100
                    #high_threshold_fixed = calculate_high_threshold(img, position, end_voxels, 0, 0.05, False)
                    noise_adjustment = sample_checker.ThresholdOutlierDetector(input_path, filename)
                    print("Starting noise variations")
                    for noise_ratio in range(0, 100, 25):
                        noisy_image = noise_adjustment.generate_noisy_image(noise_ratio/100)
                        position = testing_knee(noisy_image, log_hist=True)
                        print(output_path+filename.split('.')[0]+"noise"+str(noise_ratio)+".tif")
                        try:
                            save_hist(noisy_image, position, end_voxels, output_path+filename.split('.')[0]+"noise"+str(noise_ratio)+"Hist.tif")
                        except Exception as err:
                            print(err)
                            print("Histogram saving failed for", filename + " noise"+str(noise_ratio))
                        for option in [True, False]:
                            try:
                                high_threshold_stretching, runtime = calculate_high_threshold(noisy_image, position, end_voxels, 0, resolution, True, option)
                                if option:
                                    total_time_per_option[0] += runtime
                                    print("\tFor sample: " + filename + " with " + str(noise_ratio) + " noise "
                                          + str(runtime) + " seconds were taken")
                                else:
                                    total_time_per_option[1] += runtime
                                    print("\tFor sample: " + filename + " with " + str(noise_ratio) + " noise " + str(
                                        runtime) + " seconds were taken")
                                valid_threshold, threshold_voxels, otsu_voxels = noise_adjustment.outlierDetection(high_threshold_stretching)
                                noisy_image_results = noisy_image_results + "\tEfficient: " + str(option) + "Noise Ratio: " + \
                                                      str(noise_ratio/10) + " Sufficient Voxels? " + str(valid_threshold) + \
                                                      " Threshold Voxels: " + str(threshold_voxels) + " Otsu Voxels: " + str(otsu_voxels) + "\n"
                                print("Noise variation completed for sample", filename, "ratio at", noise_ratio/10)
                            except Exception as error:
                                failed_variants = failed_variants + "Sample: " + filename + " Noise: " + str(noise_ratio) + "\n"
                                print(error)

                    #high_threshold = 60
                    high_thresholds.append([high_threshold_stretching])
                    #fixed_thresholded_img = hysteresis_thresholding_stack(img, position, high_threshold_fixed).astype('uint8')*np.max(img)
                    stretched_thresholded_img =  hysteresis_thresholding_stack(img, position, high_threshold_stretching).astype('uint8')*255
                    #print(np.max(thresholded_img))
                    #io.imshow(thresholded_img[1])
                    #plt.show()
                    #io.imshow(img[1])
                    #plt.show()
                    zero_array = np.zeros_like(stretched_thresholded_img)
                    #fixed_overlayed_img = np.stack((img, fixed_thresholded_img, zero_array), axis=-1)
                    stretched_overlayed_img = np.stack((img, stretched_thresholded_img, zero_array), axis=-1)
                    first_img = hysteresis_thresholding_stack(img, manual_Hysteresis[filename][0][0]*255, manual_Hysteresis[filename][0][1]*255).astype('uint8')*255
                    second_img = hysteresis_thresholding_stack(img, manual_Hysteresis[filename][1][0]*255, manual_Hysteresis[filename][1][1]*255).astype('uint8')*255
                    threshold_compared = np.stack((second_img, first_img, stretched_thresholded_img), axis=-1)
                    #print(overlayed_img.shape)
                    #io.imshow(overlayed_img[1])
                    #plt.show()
                    #io.imsave(output_path + '.'.join(filename.split(sep=".")[:-1]) + "density" + str(density) + "highThresh" + str(high_threshold_fixed) + "OverlayedFixed.tif", fixed_overlayed_img)
                    #io.imsave(output_path + '.'.join(filename.split(sep=".")[:-1]) + "density" + str(density) + "resolution" + str(resolution*100) + "highThresh" + str(high_threshold_stretching) + "Overlayed.tif", stretched_overlayed_img)
                    #io.imsave(output_path + '.'.join(filename.split(sep=".")[:-1]) + "density" + str(density) + "resolution" + str(resolution * 100) + "highThresh" + str(high_threshold_stretching) + "Compared.tif", threshold_compared)
                    output_of_results = output_of_results + "Density of " + str(density) + " provides a stretched threshold of " + str(high_threshold_stretching) + " at a resolution of " + str(resolution*100) + "\n"
            thresholds_per_sample[filename] = high_thresholds
            print("Manual high thresholds. First", manual_Hysteresis[filename][0][1]*255, " Second", manual_Hysteresis[filename][1][1]*255)
        #histogram_density(img, position)
        '''
        log_low, log_high = determine_hysteresis_thresholds(img, moving_average_frame=20, cut_off_slope=3, log_value=True)
        results = hist_threshold_differences(filename, low, position, img)
        log_results = hist_threshold_differences(filename, log_low, log_position, img)
        complete_results[filename] = {"Normal":results, "Log":log_results}'''
        #iterate_high(img, position)
        #position, valid = detectElbows(img)
        '''
        elbow_high = (1.0-(1.0-position/255)/2)*255
        #elbowThresh = hysteresis_thresholding_stack(img, position, elbow_high)
        #io.imsave(output_path + filename + ".tif", elbowThresh)
        valid = True
        #hist_compare(img)
        print("Hurrah ", position, " file: ", filename)
        if valid:
            results += "Sample: " + str(filename) + " Threshold: " + str(position) + " High: " + str(elbow_high) + "\n"
            results += "Low: " + str(low*255) + " High: " + str(high*255) + "\n"
            results += "Actual: " + str(manual_Hysteresis[filename][0][0]*255) + " " + str(manual_Hysteresis[filename][0][1]*255) + " " + str(manual_Hysteresis[filename][1][0]*255) + " " + str(manual_Hysteresis[filename][1][1]*255) + "\n"
        '''
        '''result = image_average(img, manual_Hysteresis[filename], False)
        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(10, 7)
        ax1.imshow(result[int(result.shape[0]/2)])
        plt.show()'''
        '''counts, centers = histogram(img, nbins=256)
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.savefig("C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\Hist.tif")
        print("Saved histogram")'''
        #img = equalize_hist(img)
        #testing_knee(img, int(elbow_high))
    print(thresholds_per_sample)

    print("Times", total_time_per_option)

    h = open(output_path + "NoiseVariationComparison.txt", "w")
    h.write(noisy_image_results)
    h.close()

    f = open(output_path + "densityVariationResults.txt", "w")
    f.write(output_of_results)
    f.close()

    g = open(output_path + "disqualified_samples.txt", "w")
    g.write(disqualified_list)
    g.close()

    m = open(output_path + "failed_samples.txt", "w")
    m.write(failed_variants)
    m.close()

    #print(complete_results)
    for key, values in complete_results.items():
        print("Sample Results", key, "\nDetermined MAE: Normal=", values["Normal"][0][0], " Log=", values["Log"][0][0], "\nKnee MAE: Normal=", values["Normal"][0][1], " Log=", values["Log"][0][1])

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

def get_high(low, ratio):
    return (255 - (255 - low)/ratio)

def parameter_thresholder(img, sample, moving_average, cutoff, low_values):
    results = []
    for low in low_values:
        high = get_high(low, 2)
        normalize_img = img / np.max(img)
        thresholded = hysteresis_thresholding_stack(normalize_img, low, high)
        manual = image_average(img)
        HystThreshold = thresholded.astype('uint8')
        mse_result = mse(manual / np.max(manual), HystThreshold)
        #copy loop from main() for where this is implemented as the three nested for loops organize the data for printing and to check if valid file

def iterate_high(img, low_threshold, step_size=5):
    maximum = np.max(img)
    normalize_img = img / maximum
    offset = int((maximum - low_threshold)/step_size) #Number of steps rounded down to nearest integer
    offset = maximum - offset*step_size + step_size #This result will be one step plus the distance different between them. In future the step_size here could be multiplied for greater offset
    step_size = -1*step_size
    steps = []
    populations = []
    print("Low Threshold:", low_threshold)
    print("Offset:", offset)
    for high_thresh in range(maximum, offset, step_size):
        thresholded = hysteresis_thresholding_stack(normalize_img, low_threshold/255, high_thresh/255)
        populations.append(thresholded.sum())
        steps.append(high_thresh)
    plt.figure(figsize=(6, 6))
    plt.plot(steps, populations, color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("The automatically calculated hysteresis thresholding values")
    plt.tight_layout()
    plt.show()

def histogram_density(img, start_point = 1, cutoff = 0.25):
    counts, centers = histogram(img, nbins=256)
    counts = counts[start_point:]
    centers = centers[start_point:]
    total_voxels = np.sum(counts)
    counter = 0
    for v in range(len(counts), 0, -1):
        counter += 1
        if np.sum(counts[v:])/total_voxels >= cutoff:
            print("Density reached at ", v)
            print(centers[v])
            print(total_voxels)
            print(check_neighbouring_voxels(v, img))
            break
    print("Number of loops: ", counter)

def check_neighbouring_voxels(current_threshold, img, decay_rate=0.1):
    #current_threshold is the max to be used for the
    total_neighbours = 0
    #print("Progress: ")
    padded_img = np.pad(img, 1)
    progress = 0
    high_intensity_threshold = (np.zeros_like(img) + 1) * current_threshold
    core_structures = np.greater_equal(img, high_intensity_threshold)
    number_of_core = core_structures.sum()
    for x in range(1, padded_img.shape[0] - 1, 1):
        for y in range(1, padded_img.shape[1] - 1, 1):
            for z in range(1, padded_img.shape[2] - 1, 1):
                if padded_img[x, y, z] >= current_threshold:
                    progress += 1
                    tracker = padded_img[x-1:x+2, y-1:y+2, z-1:z+2]
                    reduced_intesity = current_threshold - current_threshold*decay_rate
                    logical_array = (np.zeros((3, 3, 3)) + 1) * reduced_intesity
                    boolean_array = np.greater_equal(tracker, logical_array)
                    kernel = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
                    results = boolean_array * kernel
                    #print(progress/(img.shape[0] * img.shape[1] * img.shape[2]), "%")
                    total_neighbours += results.sum()
    return total_neighbours, number_of_core

def calculate_high_threshold(img, low, start_density, stop_margin, decay_rate=0.1, percentile=True, faster=True):
    '''
    - low is used to remove the low intensity and noise voxels which will greatly skew the density
    - start_density to acquire the high intensity threshold starting point. Voxels above this intensity are within this ratio
    - stop_margin is the margin of voxels gained to determine a stopping point. This should be a ratio such that voxels_retained/reference
    where reference will either be the prior voxels retained or the total voxels retained. The ratio needs to be decided
    - decay_rate is the rate at which the threshold value reduces by for each neighbouring check
    '''
    print("Calculation beginning", low)
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
            print("Starting intensity found", starting_intensity)
            break

    #Now the starting high threshold has been acquired the rest can begin
    if percentile:
        range_of_intensities = recursive_intensity_steps(low, starting_intensity, decay_rate) #This will provide a list of intensities with which to check for neighbours
        range_of_intensities.insert(0, starting_intensity)
    else:
        range_of_intensities = fixed_intensity_steps(low, starting_intensity, decay_rate)
        if len(range_of_intensities) == 0:
            return 0
        if range_of_intensities[0] != starting_intensity:
            range_of_intensities.insert(0, starting_intensity)
    array_of_structures = np.zeros_like(img)

    range_of_intensities_time_start = time.process_time()
    for i in range_of_intensities:
        compare_array = template_compare_array * i
        results = np.greater_equal(img, compare_array).astype(int)
        array_of_structures = array_of_structures + results
    range_of_intensities_time_end = time.process_time()
    print("Time taken for intensity discretisation:", range_of_intensities_time_end-range_of_intensities_time_start)

    #array_of_structures an array of ints. The voxel values are proportional to the number of intensities from range_of_intensities that the voxel is greater than/equal to
    print("Range of intensities:", range_of_intensities)
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
                    adjacency_array[z, y, x]
            for r_ele in reverse_it:
                z = reverse_it.multi_index[0]
                y = reverse_it.multi_index[1]
                x = reverse_it.multi_index[2]
                if r_ele == r and np.any(adjacency_array[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]):
                    adjacency_array[z, y, x]
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
        if change <= stop_margin:
            print(progress)
            print("Stopped at ", range_of_intensities[r])
            return
        else:
            old_voxels = new_voxels
        change_by_intensity.append(change)
        neighbour_validation_time_end = time.process_time()
        print("Time taken for intensity", range_of_intensities[r], " took", neighbour_validation_time_end-neighbour_validation_time_start)
        print("Time per repeat", time_per_repeat)
    average_of_change = sum(change_by_intensity)/(len(range_of_intensities) - 1)
    print("The average of the change in voxels is", average_of_change)
    answer = 0
    for cbi in range(0, len(change_by_intensity), 1):
        if change_by_intensity[cbi] <= average_of_change:
            print("The best threshold is at", range_of_intensities[cbi+1])
            answer = range_of_intensities[cbi+1]
            break
    print(progress)
    return answer, time_total

def fixed_intensity_steps(bottom, top, ratio):
    steps = []
    step_size = int((top - bottom) * ratio)
    if step_size == 0:
        print("Step_size is 0", top, bottom)
        return steps
    for step in range(top, bottom, -1 * step_size):
        steps.append(step)
    return steps

def recursive_intensity_steps(bottom, top, ratio):
    new_intensity = int(top * (1 - ratio)) #this will reduce top by 10%
    steps = [new_intensity]
    if int(new_intensity * (1 - ratio)) > bottom:
        results = recursive_intensity_steps(bottom, new_intensity, ratio)
        for r in results:
            steps.append(r)
    return steps

def density_evaluate(low, img):
    counts, centers = histogram(img)
    low_index = np.where(centers == low)[0][0]
    counts = counts[low_index:]
    centers = centers[low_index:]
    plt.plot(centers, counts, color='black')
    plt.show()

def hist_compare(img):
    kernel_size = np.array([img.shape[dim] // 8 for dim in range(img.ndim)])
    kernel_size[0] = img.shape[0]
    adapt_img = equalize_adapthist(img, kernel_size)
    equal_img = equalize_hist(img)
    p2, p98 = np.percentile(img, (2, 98))
    stretched_img = rescale_intensity(img, in_range=(p2, p98))
    adapt_hist = {}
    equal_hist = {}
    normal_hist = {}
    stretched_hist = {}

    adapt_hist[0], adapt_hist[1] = histogram(adapt_img, nbins=256)
    equal_hist[0], equal_hist[1] = histogram(equal_img, nbins=256)
    normal_hist[0], normal_hist[1] = histogram(img, nbins=256)
    stretched_hist[0], stretched_hist[1] = histogram(stretched_img, nbins=256)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('Histogram comparisons')

    adapt_line = testing_knee(adapt_img)
    equal_line = testing_knee(equal_img)
    normal_line = testing_knee(img)

    ax1.axvline(adapt_line, 0, 1, label='Adaptive', color="blue")
    ax2.axvline(equal_line, 0, 1, label='Equal', color="green")
    ax3.axvline(normal_line, 0, 1, label='Normal', color="red")
    print("Adaptive: ", adapt_line, " Equal: ", equal_line, " Normal: ", normal_line)

    adapt_hist[1] = np.where(adapt_hist[1][1:] != 0, np.log10(adapt_hist[1][1:]), 0)
    equal_hist[1] = np.where(equal_hist[1][1:] != 0, np.log10(equal_hist[1][1:]), 0)
    normal_hist[1] = np.where(normal_hist[1][1:] != 0, np.log10(normal_hist[1][1:]), 0)
    stretched_hist[1] = stretched_hist[1][1:]

    adapt_hist[0] = adapt_hist[0][1:]*256
    equal_hist[0] = equal_hist[0][1:]*256
    normal_hist[0] = normal_hist[0][1:]
    stretched_hist[0] = stretched_hist[0][1:]

    ax1.plot(adapt_hist[1], adapt_hist[0], color="black")
    ax2.plot(equal_hist[1], equal_hist[0], color="black")
    ax3.plot(normal_hist[1], normal_hist[0], color="black")
    ax4.plot(stretched_hist[1], stretched_hist[0], color="black")
    plt.tight_layout()
    plt.show()


def hist_average(img, movingAverageFrame = 20):
    counts, centers = histogram(img, nbins=256)
    counts = counts[1:]
    centers = centers[1:]
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Histogram Average Compare')
    ax1.plot(centers, counts, color="black")
    counts = np.where(counts != 0, np.log2(counts), 0)
    ax2.plot(centers, counts, color="black")
    df = pd.DataFrame(counts)
    moving_average = df.rolling(movingAverageFrame, center=True).mean()
    moving_average_array = moving_average.T.to_numpy()
    counts = moving_average_array[0]
    print("Average Array:", moving_average_array[0][10])
    # print(movingAverageArray[0].shape, centers.shape)

    #plt.figure(figsize=(6, 4))
    ax3.plot(centers, counts, color='black')
    #plt.xlabel("Intensity")
    #plt.ylabel("Count")
    #plt.title("The automatically calculated hysteresis thresholding values")
    plt.tight_layout()
    plt.show()

def testing_knee(img, cutoff = 1, log_hist=False):
    print("Histogram for knee")
    counts, centers = histogram(img, nbins=256)
    counts = counts[cutoff:]
    #print("Final Counts: ", counts[-20:-1])
    if log_hist:
        counts = np.where(counts != 0, np.log10(counts), 0)
    centers = centers[cutoff:]
    #print(centers.shape)
    '''plt.figure(figsize=(6, 6))
    plt.plot(centers, counts, color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("Histogram")
    plt.tight_layout()
    plt.show()'''

    locator = KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing")
    print("Norm Knee", locator.norm_knee*255)
    print("Standard Knee", locator.knee)
    knee = int(locator.knee)
    #print("knees: ", locator.all_knees, " knee heights: ", locator.all_knees_y)


    #locator.plot_knee()
    #plt.show()
    #knee = int(locator.norm_knee*255)

    return knee

def image_compare():
    raw_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    input_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisDatatoReview\\"
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    samples = [f.split(sep="density")[0]+".tif" for f in listdir(input_path) if isfile(join(input_path, f))]
    results = {}
    for s in range(0, len(samples), 1):
        first_mse_results = []
        second_mse_results = []
        average_mse_results = []
        manual_diffs = []
        if isfile(join(raw_path, samples[s])):
            raw_img = io.imread(raw_path + samples[s])
            estimate_image = io.imread(input_path + images[s]).astype('uint8')
            parameters = manual_Hysteresis[samples[s]]
            first_img = hysteresis_thresholding_stack(raw_img, parameters[0][0]*255, parameters[0][1]*255).astype('uint8')
            second_img = hysteresis_thresholding_stack(raw_img, parameters[1][0]*255, parameters[1][1]*255).astype('uint8')
            average_img = image_average(raw_img, parameters).astype('uint8')
            first_mse_result = (mse(estimate_image/np.max(estimate_image), first_img/np.max(first_img)))
            second_mse_result = (mse(estimate_image/np.max(estimate_image), second_img/np.max(second_img)))
            average_mse_result = (mse(estimate_image/np.max(estimate_image), average_img/np.max(average_img)))
            manual_diff = (mse(first_img/np.max(first_img), second_img/np.max(second_img)))
            first_mse_results.append(first_mse_result)
            second_mse_results.append(second_mse_result)
            average_mse_results.append(average_mse_result)
            manual_diffs.append(manual_diff)
            print("Results for", images[s], " first MSE result", first_mse_result, " second MSE result", second_mse_result, " average MSE results", average_mse_result, " error between manual", manual_diff)
        results[images[s]] = [first_mse_results, second_mse_results, average_mse_results, manual_diffs]
    print(results)



def main():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Input Data\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Output Data\\"
    threshold_record_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Output Data\\Record.txt"
    thresholded_compare_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Manual Thresh\\"
    if exists(threshold_record_path):
        record = open(threshold_record_path, 'a')
    else:
        record = open(threshold_record_path, 'w')
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    print(images)
    parameter_variations = [[10, 20],[2, 4]]

    i_dict = {}
    for i in images:
        if i in listdir(thresholded_compare_path):
            manual_thresh = image_average(io.imread(input_path+i, ))
            m_dict = {}
            for m in parameter_variations[0]:
                c_dict = {}
                for c in parameter_variations[1]:
                    HystThreshold = thresholding(i, input_path, output_path, m, c, record)
                    HystThreshold = HystThreshold.astype('uint8')
                    mse_result = mse(manual_thresh/np.max(manual_thresh), HystThreshold)
                    #ssim_Result = ssim(manual_thresh, HystThreshold)
                    c_dict[str(c)] = mse_result
                m_dict[str(m)] = c_dict
            i_dict[i] = m_dict
    comparison_results = np.zeros([len(parameter_variations[0]), len(parameter_variations[1])])
    num_files = len(list(i_dict))
    for files, values in i_dict.items():
        for m_index, m_values in values.items():
            for c_index, c_values in m_values.items():
                comparison_results[parameter_variations[0].index(int(m_index)), parameter_variations[1].index(int(c_index))] += c_values
    comparison_results = comparison_results / num_files
    print(i_dict)
    print(comparison_results)
    for m in parameter_variations[0]:
        for c in parameter_variations[1]:
            print("Value for m", m, "and c", c, " = ", comparison_results[parameter_variations[0].index(int(m)), parameter_variations[1].index(int(c))])
    print("C Average", np.mean(comparison_results, axis=0))
    print("M Average", np.mean(comparison_results, axis=1))


    record.close()
    return

def image_MAE(image, manual_parameters, reference_image, reference_index):
    first_image = apply_hysteresis_threshold(image, manual_parameters[0][0]*255, manual_parameters[0][1]*255) #Image from my (Richard) parameters
    second_image = apply_hysteresis_threshold(image, manual_parameters[1][0]*255, manual_parameters[1][1]*255) #Image from Rensu's parameters
    automatic_threshold = reference_image[:,:,:,reference_index]
    automatic_threshold = automatic_threshold/np.max(automatic_threshold)
    first_image_error = np.average(np.abs(first_image - automatic_threshold))
    second_image_error = np.average(np.abs(second_image - automatic_threshold))
    #print("First Error", np.sum(np.abs(second_image - automatic_threshold))/max(np.sum(second_image), np.sum(automatic_threshold)))
    #print("Original First Error", first_image_error)
    return first_image_error, second_image_error


def hist_threshold_differences(sample_name, determined_low, kneedle_low, image = False):
    determined_high = (0.5 + determined_low*0.5)
    determined_low = determined_low
    kneedle_high = (0.5 + kneedle_low/255 * 0.5)*255
    #print("Determined Low:", determined_low*255, "Determined high:", determined_high*255)
    #print("Kneedle Low:", kneedle_low, "Kneedle high:", kneedle_high)
    manual_values = manual_Hysteresis[sample_name]
    pairs = {}
    for m in range(0, len(manual_values), 1):
        for n in range(m+1, len(manual_values), 1):
            pairs[str(m) + str(n)] = [manual_values[m], manual_values[n]]
    mean_pairs = []
    for k, p in pairs.items():
        distance = math.sqrt((p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
        mae = (abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1])) / 2
        pairs[k].append([distance, mae])
        mean_pairs.append([(p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2])
    calc_differences = []
    for manual in manual_values:
        determined_distance = math.sqrt((manual[0] - determined_low) ** 2 + (manual[1] - determined_high) ** 2)
        kneedle_distance = math.sqrt((manual[0] - kneedle_low/255) ** 2 + (manual[1] - kneedle_high/255) ** 2)
        determined_mae = (abs(manual[0] - determined_low) + abs(manual[1] - determined_high))/2
        kneedle_mae = (abs(manual[0] - kneedle_low/255) + abs(manual[1] - kneedle_high/255))/2
        calc_differences.append([determined_mae, kneedle_mae])
    #print(pairs)
    #print(calc_differences)
    #print(mean_pairs)
    MAE_results = []
    for m_pairs in mean_pairs:
        mean_kneedle_mae = math.sqrt((m_pairs[0] - kneedle_low/255) ** 2 + (m_pairs[1] - kneedle_high/255) ** 2)
        mean_determined_mae = (abs(m_pairs[0] - kneedle_low/255) + abs(m_pairs[1] - kneedle_high/255))/2
        MAE_results.append([mean_determined_mae, mean_kneedle_mae])
        #print("Mean Determined Dist:", math.sqrt((m_pairs[0] - determined_low) ** 2 + (m_pairs[1] - determined_high) ** 2))
        #print("Mean Determined MAE:", (abs(m_pairs[0] - determined_low) + abs(m_pairs[1] - determined_high))/2)
        #print("Mean Kneedle Dist:", mean_kneedle_mae)
        #print("Mean Kneedle MAE:", mean_determined_mae)
    #counts, centers = histogram(image, nbins=256)
    '''for k, pair in pairs.items():
        # remove 'black'
        start = int(min(pair[0][1], pair[1][1])*255 - 40)
        print(start)
        count = counts[1:]
        center = centers[1:]
        print(center[0])
        print(len(count), len(center))
        plt.figure(figsize=(6, 4))
        plt.plot(center, count, color='black')
        #plt.axvline(pair[0][1]*255, 0, 1, label='FirstHigh', color="red")
        #plt.axvline(pair[1][1]*255, 0, 1, label='SecondHigh', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.show()'''
    return MAE_results

def image_average(input_image, parameters, thresh_type=False):
    thresholded_images = []
    value1 = 0
    value2 = 0
    for p in parameters:
        value1 += p[0]
        value2 += p[1]
        if thresh_type:
            thresholded_images.append(adaptive_threshold_stack(input_image, p[0], p[1]))
        else:
            thresholded_images.append(hysteresis_thresholding_stack(input_image, p[0]*255, p[1]*255).astype(int))
    if thresh_type:
        thresholded_images.append(adaptive_threshold_stack(input_image, value1/len(parameters), value2/len(parameters)))
    else:
        thresholded_images.append(hysteresis_thresholding_stack(input_image, value1/len(parameters), value2/len(parameters)).astype(int))
    average_image = np.mean(np.array(thresholded_images), axis=0)
    return np.round(average_image)

def adaptive_threshold_stack(input_image, block_size = 100, constant = -30):
    if block_size % 2 != 1:
        block_size += 1

    thresholded = []
    for i in range(input_image.shape[0]):
        thresholded.append(cv2.adaptiveThreshold(input_image[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)) # ADAPTIVE_THRESH_MEAN_C
    return np.array(thresholded)

def thresholding(i, input_path, output_path, moving_average_frame, cut_off_slope, record_name):
    img = io.imread(input_path + i)
    print(type(img))
    if len(img.shape) <= 3:
        print(i)
        filename = '.'.join(i.split(sep=".")[:-1]) + "m" + str(moving_average_frame) + "c" + str(
            cut_off_slope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determine_hysteresis_thresholds(img=img, outputPath=output_path, moving_average_frame=moving_average_frame,cut_off_slope=cut_off_slope)
        normalize_img = img / np.max(img)
        thresholded = hysteresis_thresholding_stack(normalize_img, low, high)
        #record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        #io.imsave(output_path + filename, thresholded)
        return thresholded

def thresholdingLoop(i, input_path, output_path, moving_average_frame, cut_off_slope, record_name):
    img = io.imread(i)
    if len(img.shape) <= 3:
        filename = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1]) + "m" + str(moving_average_frame) + "c" + str(
            cut_off_slope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determine_hysteresis_thresholds(img=img, outputPath=output_path, moving_average_frame=moving_average_frame,
                                                  cut_off_slope=cut_off_slope)
        normalize_img = img / np.max(img)
        thresholded = hysteresis_thresholding_stack(normalize_img, low, high)
        record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        io.imsave(output_path + filename, thresholded)
    else:
        print("TIME", img.shape)
        threshold_timepoints = []
        print("Number of images: ", img.shape[0])
        fileFirstHalf = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1])
        fileEndHalf = "m" + str(moving_average_frame) + "c" + str(cut_off_slope) + ".tif"
        for t in range(img.shape[0]):
            print("Index: ", t)
            #hist_name="Hist of " + fileFirstHalf + "t" + str(t) + fileEndHalf + ".png",
            low, high = determine_hysteresis_thresholds(img=img[t], outputPath=output_path, moving_average_frame=moving_average_frame,
                                                      cut_off_slope=cut_off_slope)
            normalize_img = img[t] / np.max(img[t])
            threshold_timepoints.append(hysteresis_thresholding_stack(normalize_img, low, high))
            record_name.write(fileFirstHalf + "t" + str(t) + fileEndHalf + " low: " + str(low * 256) + " high: " + str(high * 256) + "\n")
        thresholded = np.stack(threshold_timepoints)
        #print(type(thresholded))
        filename = fileFirstHalf + fileEndHalf
        io.imsave(output_path + filename, thresholded)
        # NB!!!!!!! Need to Try catch and fix dimensions for FiJi (time + z combined!!)


def hysteresis_thresholding_stack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def determine_hysteresis_thresholds(img, outputPath=None, hist_name=None, bins=256, moving_average_frame=20, cut_off_slope=2, log_value=False, highVal=0.95): #This function is from Rensu's MEL (Make sure to reference)
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]
    #counts = np.where(counts != 0, np.log10(counts), 0)
    df = pd.DataFrame(counts)
    movingAverage = df.rolling(moving_average_frame, center=True).mean()
    movingAverageArray = movingAverage.T.to_numpy()
    counts = movingAverageArray[0]
    if log_value:
        counts = np.where(movingAverageArray[0] != 0, np.log10(movingAverageArray[0]), 0)
    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0

    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        if counts[i-10]/counts[i+10] >= cut_off_slope:
              useIntensityLow = i
              print("Low intensity to be used: ", useIntensityLow)
              print("High intensity to be used: ", ((bins-useIntensityLow)*(useIntensityLow/bins) + useIntensityLow))

              break

    print(outputPath)
    '''plt.figure(figsize=(6, 6))
    plt.plot(centers, counts, color='black')
    plt.axvline(useIntensityLow, 0, 1, label='Low', color="red")
    plt.axvline((1.0 - (1.0 - useIntensityLow / bins) / 2) * bins, 0, 1, label='High', color="blue")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("The automatically calculated hysteresis thresholding values")
    plt.tight_layout()
    plt.show()'''
    '''
    if outputPath != None and hist_name != None:
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.axvline(useIntensityLow, 0, 1, label='Low', color="red")
        plt.axvline((1.0-(1.0-useIntensityLow/bins)/2)*bins, 0, 1, label='High', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        outputPath = outputPath + "\\" + hist_name
        plt.savefig(outputPath)
        print("Saved histogram")'''

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/4))

def detectElbows(img, movingAverageFrame=20, elbowSize=10):
    #The gradient margin can get big (28%) thus the selected points are not even close to being perpendicular. A line of values between points needs to be generated and then check for interception. If it intercepts between two points then pick the closest or rightmost
    counts, centers = histogram(img, nbins=256)
    #remove black
    counts = np.log10(counts[1:])
    centers = centers[1:]
    # [intensity, pixel_count]
    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()
    print(movingAverage.min(axis = 0))
    startIntensity = 10
    #print(len(movingAverage[0]))
    diagnose = {}
    distPoints = []
    perpendicularPoints = []
    for i in range(len(movingAverage[0]) - elbowSize - 1, elbowSize + 1, -1):
        rightmost = movingAverage[0][i + elbowSize]
        leftmost = movingAverage[0][i - elbowSize]
        gradient = (rightmost - leftmost)/(2*elbowSize)
        centerIntensity = abs(rightmost-leftmost)/2
        try:
            if gradient > 0:
                best_position = i+1
                best_grad = 0
                first = True
                for p in range(i+1, i+elbowSize, 1):
                    perp_grad = elbowGradientCheck(movingAverage[0][p], centerIntensity, p, i)
                    grad_diff = abs(perp_grad + gradient)
                    if first:
                        first = False
                        best_grad = perp_grad
                        best_position = p
                    else:
                        if grad_diff < abs(best_grad + gradient):
                            best_position = p
                            best_grad = perp_grad
                perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [best_position, movingAverage[0][best_position], best_grad]])
                distance = math.sqrt((best_position-i)**2 + (movingAverage[0][best_position] - centerIntensity)**2)
                distPoints.append([best_position, distance])

            elif gradient < 0:
                best_position = i + 1
                best_grad = 0
                first = True
                for p in range(i-1, i-elbowSize, -1):
                    perp_grad = elbowGradientCheck(movingAverage[0][p], centerIntensity, p, i)
                    grad_diff = abs(perp_grad + gradient)
                    if first:
                        first = False
                        best_grad = perp_grad
                        best_position = p
                    else:
                        if grad_diff < abs(best_grad + gradient):
                            best_position = p
                            best_grad = perp_grad
                perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [best_position, movingAverage[0][best_position], best_grad]])
                distance = math.sqrt((best_position - i) ** 2 + (movingAverage[0][best_position] - centerIntensity) ** 2)
                distPoints.append([best_position, distance])
            else:
                if centerIntensity > movingAverage[0][i]:
                    perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [i, movingAverage[0][i]]])
                    distPoints.append([i, centerIntensity - movingAverage[0][i]])
        except Exception as e:
            print(e)
            print("")

    if distPoints:
        #This is a cheat since I know that distPoints will be of a list type so this should always work
        distances = np.array(distPoints)
        #print(distances[:, 1])
        maximum_values = np.argmax(distances[:, 1])
        #print("Yay", maximum_values)
        #print(distances[maximum_values])
        print(distPoints[maximum_values])
        print("PerpPoints", perpendicularPoints[maximum_values])
        orig_pos = perpendicularPoints[maximum_values][0][0] + elbowSize + 1
        print(movingAverage[0][10:orig_pos])
        movingAverageArray = movingAverage.T.to_numpy()
        print("Average Array:", movingAverageArray[0][10])
        # print(movingAverageArray[0].shape, centers.shape)
        '''plt.figure(figsize=(6, 4))
        plt.plot(centers, movingAverageArray[0], color='black')
        plt.xlabel("Intensity")
        plt.ylabel("AverageCount")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.show()'''
        return distances[maximum_values][0].astype(int), True
    return 0, False

def elbowGradientCheck(intensity1, intensity2, position1, position2):
    if (position1 - position2) == 0:
        print("Divide by zero warning:", position1)
    try:
        return (intensity1 - intensity2) / (position1 - position2)

    except Exception as e:
        print("Ydelta:", (intensity1 - intensity2), "Xdelta:", (position1 - position2))
        return False

def preview_hysteresis_high(img, low):
    total_array = np.zeros_like(img)
    print("Building Preview")
    for i in range(np.max(img), low, -1):
        print("Intensity", i)
        threshold = hysteresis_thresholding_stack(img, low, i)
        total_array += threshold
    total_array[total_array > 0] += low #low offset for all non-zero elements so that the voxels can be correctly evaluated by intensity value
    return total_array



if __name__ == "__main__":
    #main()
    time_start = time.process_time()
    testing()
    print("In total took ", time.process_time() - time_start)
    #image_compare()
    #test_saved_images("Overlayed")