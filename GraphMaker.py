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
from scipy import interpolate
import tifffile
from knee_locator import KneeLocator
from sklearn import metrics

manual_Hysteresis = {"CCCP_1C=0.tif": [[0.1, 0.408], [0.1, 0.25]], "CCCP_1C=1.tif": [[0.116, 0.373], [0.09, 0.22]],
                     "CCCP_2C=0.tif": [[0.107, 0.293], [0.09, 0.2]], "CCCP_2C=1.tif": [[0.09, 0.372], [0.08, 0.15]],
                     "CCCP+Baf_2C=0.tif": [[0.093, 0.279], [0.1, 0.17]], "CCCP+Baf_2C=1.tif": [[0.098, 0.39], [0.1, 0.35]],
                     "Con_1C=0.tif": [[0.197, 0.559], [0.14, 0.18]], "Con_1C=2.tif": [[0.168, 0.308], [0.11, 0.2]],
                     "Con_2C=0.tif": [[0.219, 0.566], [0.19, 0.31]], "Con_2C=2.tif": [[0.137, 0.363], [0.13, 0.23]],
                     "HML+C+B_2C=0.tif": [[0.102, 0.55], [0.14, 0.31]], "HML+C+B_2C=1.tif": [[0.09, 0.253], [0.09, 0.18]],
                     "HML+C+B_2C=2.tif": [[0.114, 0.477], [0.11, 0.31]], "LML+C+B_1C=0.tif": [[0.09, 0.152], [0.05, 0.1]],
                     "LML+C+B_1C=1.tif": [[0.102, 0.232], [0.07, 0.15]], "LML+C+B_1C=2.tif": [[0.034, 0.097], [0.024, 0.1]]}

def scrape_json(input_path):
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    json_files = [f for f in files if ".json" in f]
    data_files = []
    for j in json_files:
        if "Complete" in j:
            data_files = input_path + j
            break
        else:
            data_files.append(input_path + j)
    #This way if there is incomplete data then it can still be worked with
    complete_results = {}
    if type(data_files) is list:
        for d in data_files:
            f = open(d)
            data = json.load(f)
            complete_results[d.split("Results")[0]] = data
            f.close()
    else:
        f = open(data_files)
        complete_results = json.load(f)
        f.close()
    return complete_results

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

def get_intensity_range(img, low, density, prec):
    voxels_by_intensity, intensities = histogram(img, nbins=256)
    low_index = np.where(intensities == low)[0][0]
    voxels_by_intensity = voxels_by_intensity[low_index:]
    intensities = intensities[low_index:]
    total_population = voxels_by_intensity.sum()
    initial_density = int(total_population * density/100)
    starting_intensity = intensities[-1]
    '''print("------------------------------------------------------")
    print("Total Voxels:", total_population)
    print("Starting Intensity:", starting_intensity)
    print("Low Intensity:", low)
    print("Initial Density:", initial_density, " Percentage ", density)
    print("Length of voxels per intensities", len(voxels_by_intensity), "Length of intensities", len(intensities))'''
    for intensity in range(len(voxels_by_intensity) - 1, 0, -1):
        density_found = np.sum(voxels_by_intensity[intensity:])
        #print("Intensity at", intensities[intensity])
        #print("Density Found:", density_found)
        if density_found >= initial_density:
            starting_intensity = intensities[intensity]
            #print("Starting Intensity Found:", starting_intensity)
            break
    #print("------------------------------------------------------")
    intensity_range = recursive_intensity_steps(low, starting_intensity, prec)
    intensity_range.insert(0, int(starting_intensity))
    return intensity_range

def collate_data(input_paths, image_paths=None):
    '''image_paths will be a dictionary relating the input image directory to the json input directory.
    This is to construct the intensity value range which was not saved erroneously'''
    null_list = []
    sample_specific_metrics = {}
    sample_variant_metrics = []
    if type(input_paths) is not list:
        input_paths = [input_paths]
    for i in input_paths:
        json_result = scrape_json(i)
        #print(json_result)
        for sample, s_metrics in json_result.items():
            img = None
            sample_specific_metrics[sample] = {}
            threshold_values = s_metrics["All Filter Thresh"]
            if image_paths is not None:
                if i in list(image_paths):
                    image_path = image_paths[i] + sample.split("\\")[-1].split(".")[0] + "Noise000.tif"
                    img = io.imread(image_path)
                    sample_specific_metrics[sample]["Image Max"] = int(np.max(img))
            for s, s_val in s_metrics.items():
                if s != "Sample Variants" and s != "All Filter Thresh":
                    sample_specific_metrics[sample][s] = s_val
                if s == "Sample Variants":
                    for v in s_val:  # This will step through the list of variants
                        variants_results = {}
                        for var, var_val in v.items(): #This is the loop to step through the variant keys and values
                            '''The reason for this loop is that some metrics need to be repaired as they were incorrectly recorded'''
                            if var == "Filter Type" and var_val is None:
                                distances = {}
                                for thr, thr_val in threshold_values.items():
                                    distances[thr] = abs(thr_val - v["Filter_Thresh"])
                                    if thr_val == v["Filter_Thresh"]:
                                        variants_results[var] = thr
                                        break
                                if var_val is None:
                                    variants_results[var] = min(distances, key=distances.get)
                            elif var == "Voxels per intensity" and type(var_val) is list:
                                converted_values = []
                                if len(var_val) > 1:
                                    for l in var_val:
                                        converted_values.append(float(l))
                                    variants_results[var] = converted_values
                                elif len(var_val) == 1:
                                    if type(var_val[0]) is list:
                                        for l in var_val[0]:
                                            converted_values.append(float(l))
                                        variants_results[var] = converted_values
                                else:
                                    variants_results[var] = []
                            elif var == "Voxels changes" and type(var_val) is list:
                                converted_values = []
                                if len(var_val) > 1:
                                    for l in var_val:
                                        converted_values.append(float(l))
                                    variants_results[var] = converted_values
                                elif len(var_val) == 1:
                                    if type(var_val[0]) is list:
                                        for l in var_val[0]:
                                            converted_values.append(float(l))
                                        variants_results[var] = converted_values
                                else:
                                    variants_results[var] = []
                            elif var == "Intensity Range" and var_val is None:
                                if img is not None:
                                    variants_results[var] = get_intensity_range(img, v["Low_Thresh"], v["Starting Density"], v["Precision"])
                                else:
                                    variants_results[var] = None
                            else:
                                variants_results[var] = var_val
                        variants_results["Sample"] = sample
                        sample_variant_metrics.append(variants_results)
    return sample_specific_metrics, sample_variant_metrics

def save_data(save_path, sample_specific_metrics, sample_variant_metrics):
    final_metrics = {}
    final_metrics["Specifics"] = sample_specific_metrics
    by_sample = []
    by_filter = {"Otsu":[], "Li":[], "Yen":[], "Triangle":[], "Mean":[]}
    by_prec = {0.02:[], 0.04:[], 0.06:[], 0.08:[], 0.1:[], 0.12:[], 0.14:[]}
    by_density = {1:[], 6:[], 11:[], 16:[], 21:[]}
    by_intensity = []
    by_error = []
    for sam_var in sample_variant_metrics:
        by_sample.append(sam_var)
        by_filter[sam_var["Filter Type"]].append({"Low_Thresh":sam_var["Low_Thresh"], "Filter_Thresh":sam_var["Filter_Thresh"], "Sample":sam_var["Sample"]})
        by_prec[sam_var["Precision"]].append({"Low_Thresh":sam_var["Low_Thresh"], "High_Thesh":sam_var["High_Thresh"],
                                              "Starting Density":sam_var["Starting Density"], "Sample":sam_var["Sample"]})
        by_density[sam_var["Starting Density"]].append({"Low_Thresh":sam_var["Low_Thresh"], "High_Thesh":sam_var["High_Thresh"],
                                                        "Precision":sam_var["Precision"], "Sample":sam_var["Sample"]})
        by_intensity.append({"High_Thesh":sam_var["High_Thresh"], "Intensity Range":sam_var["Intensity Range"],
                             "Voxels per intensity":sam_var["Voxels per intensity"], "Voxel Changes":sam_var["Voxels changes"],
                             "Total Voxels":sample_specific_metrics[sam_var["Sample"]]["Total Voxels"],
                             "Precision":sam_var["Precision"], "Starting Density":sam_var["Starting Density"], "Filter Type":sam_var["Filter Type"], "Sample":sam_var["Sample"]})
        by_error.append({"Sample":sam_var["Sample"], "Richard Error":sam_var["Richard Error"], "Rensu Error":sam_var["Rensu Error"],
                         "Filter Type":sam_var["Filter Type"], "Starting Density":sam_var["Starting Density"], "Precision":sam_var["Precision"]})
    final_metrics["All Metrics"] = by_sample
    final_metrics["Filter Specific"] = by_filter
    final_metrics["Prec Specific"] = by_prec
    final_metrics["Density Specific"] = by_density
    final_metrics["Intensity Metrics"] = by_intensity
    final_metrics["Error Metrics"] = by_error
    with open(save_path + 'CompleteResults.json', 'w') as j:
        json.dump(final_metrics, j)
        print("Saved")

def save_data2(save_path, sample_specific_metrics, sample_variant_metrics):
    final_metrics = {}
    final_metrics["Specifics"] = sample_specific_metrics
    by_sample = []
    by_filter = {"Otsu":[], "Li":[], "Yen":[], "Triangle":[], "Mean":[]}
    range_of_prec = range(1, 20)
    by_prec = {}
    for p in range_of_prec:
        by_prec[p/100] = []
    by_density = {1:[], 6:[], 11:[], 16:[], 21:[]}
    by_intensity = []
    by_error = []
    for sam_var in sample_variant_metrics:
        by_sample.append(sam_var)
        by_filter[sam_var["Filter Type"]].append({"Low_Thresh": sam_var["Low_Thresh"], "Filter_Thresh": sam_var["Filter_Thresh"], "Sample":sam_var["Sample"]})
        by_prec[sam_var["Precision"]].append({"Low_Thresh": sam_var["Low_Thresh"], "High_Thesh": {"Unlooped - Fixed": sam_var["Unlooped - Fixed - High_Thresh"],
                                    "Unlooped - AdjDensity": sam_var["Unlooped - AdjDensity - High_Thresh"],
                                    "Unlooped - FixedDensity": sam_var["Unlooped - FixedDensity - High_Thresh"],
                                    "Looped - Fixed": sam_var["Looped - Fixed - High_Thresh"], "Looped - AdjDensity": sam_var["Looped - AdjDensity - High_Thresh"],
                                    "Looped - FixedDensity": sam_var["Looped - FixedDensity - High_Thresh"]},
                                              "Starting Density": sam_var["Starting Density"], "Sample": sam_var["Sample"]})
        by_density[sam_var["Starting Density"]].append({"Low_Thresh": sam_var["Low_Thresh"],
                                    "High_Thesh":{"Unlooped - Fixed": sam_var["Unlooped - Fixed - High_Thresh"],
                                    "Unlooped - AdjDensity": sam_var["Unlooped - AdjDensity - High_Thresh"],
                                    "Unlooped - FixedDensity": sam_var["Unlooped - FixedDensity - High_Thresh"],
                                    "Looped - Fixed": sam_var["Looped - Fixed - High_Thresh"],
                                    "Looped - AdjDensity": sam_var["Looped - AdjDensity - High_Thresh"],
                                    "Looped - FixedDensity": sam_var["Looped - FixedDensity - High_Thresh"]},
                                    "Precision": sam_var["Precision"], "Sample": sam_var["Sample"]})

        by_intensity.append({"High_Thesh": {"Unlooped - Fixed": sam_var["Unlooped - Fixed - High_Thresh"],
                                    "Unlooped - AdjDensity": sam_var["Unlooped - AdjDensity - High_Thresh"],
                                    "Unlooped - FixedDensity": sam_var["Unlooped - FixedDensity - High_Thresh"],
                                    "Looped - Fixed": sam_var["Looped - Fixed - High_Thresh"],
                                    "Looped - AdjDensity": sam_var["Looped - AdjDensity - High_Thresh"],
                                    "Looped - FixedDensity": sam_var["Looped - FixedDensity - High_Thresh"]},
                             "Intensity Range": {"Fixed": sam_var["Fixed - Intensity Range"], "AdjDensity": sam_var["AdjDensity - Intensity Range"],
                                                "FixedDensity": sam_var["FixedDensity - Intensity Range"]},
                             "Voxels per intensity": {"Unlooped - Fixed": sam_var["Unlooped - Fixed - Voxels per intensity"],
                                                     "Unlooped - AdjDensity": sam_var["Unlooped - AdjDensity - Voxels per intensity"],
                                                     "Unlooped - FixedDensity": sam_var["Unlooped - FixedDensity - Voxels per intensity"],
                                                     "Looped - Fixed": sam_var["Looped - Fixed - Voxels per intensity"],
                                                     "Looped - AdjDensity": sam_var["Looped - AdjDensity - Voxels per intensity"],
                                                     "Looped - FixedDensity": sam_var["Looped - FixedDensity - Voxels per intensity"]},
                             "Scaled Voxels per intensity": {"Unlooped - Fixed": sam_var["Unlooped - Fixed - Scaled Voxels per intensity"],
                                                      "Unlooped - AdjDensity": sam_var["Unlooped - AdjDensity - Scaled Voxels per intensity"],
                                                      "Unlooped - FixedDensity": sam_var["Unlooped - FixedDensity - Scaled Voxels per intensity"],
                                                      "Looped - Fixed": sam_var["Looped - Fixed - Scaled Voxels per intensity"],
                                                      "Looped - AdjDensity": sam_var["Looped - AdjDensity - Scaled Voxels per intensity"],
                                                      "Looped - FixedDensity": sam_var["Looped - FixedDensity - Scaled Voxels per intensity"]},
                             "Voxel Changes":{"Unlooped - Fixed":sam_var["Unlooped - Fixed - Voxels changes"],
                                              "Unlooped - AdjDensity":sam_var["Unlooped - AdjDensity - Voxels changes"],
                                              "Unlooped - FixedDensity":sam_var["Unlooped - FixedDensity - Voxels changes"],
                                              "Looped - Fixed":sam_var["Looped - Fixed - Voxels changes"],
                                              "Looped - AdjDensity":sam_var["Looped - AdjDensity - Voxels changes"],
                                              "Looped - FixedDensity":sam_var["Looped - FixedDensity - Voxels changes"]},
                             "Total Voxels": sample_specific_metrics[sam_var["Sample"]]["Total Voxels"],
                             "Precision": sam_var["Precision"], "Starting Density": sam_var["Starting Density"], "Filter Type": sam_var["Filter Type"],
                             "Sample": sam_var["Sample"], "Low_Thresh": sam_var["Low_Thresh"], "Image Max": sample_specific_metrics[sam_var["Sample"]]["Image Max"]})

        by_error.append({"Sample": sam_var["Sample"], "Richard Error": sam_var["Richard Error"], "Rensu Error": sam_var["Rensu Error"],
                         "Filter Type": sam_var["Filter Type"], "Starting Density": sam_var["Starting Density"], "Precision": sam_var["Precision"]})
    final_metrics["All Metrics"] = by_sample
    final_metrics["Filter Specific"] = by_filter
    final_metrics["Prec Specific"] = by_prec
    final_metrics["Density Specific"] = by_density
    final_metrics["Intensity Metrics"] = by_intensity
    final_metrics["Error Metrics"] = by_error
    with open(save_path + 'CompleteResults.json', 'w') as j:
        json.dump(final_metrics, j)
        print("Saved")

def load_data(saved_path):
    f = open(saved_path + "CompleteResults.json")
    data = json.load(f)
    f.close()
    specifics = data["Specifics"]
    all_metrics = data["All Metrics"]  # List of Dictionaries
    by_filter = data["Filter Specific"]  # Dictionary of lists of dictionaries
    by_prec = data["Prec Specific"]  # Dictionary of lists of dictionaries
    by_density = data["Density Specific"]  # Dictionary of lists of dictionaries
    by_intensity = data["Intensity Metrics"]  # List of dictionaries
    by_error = data["Error Metrics"]  # List of dictionaries
    return {"Specifics":specifics, "All":all_metrics, "Filter":by_filter, "Precision":by_prec, "Density":by_density, "Intensity":by_intensity, "Error":by_error}

def filter_reduce(filter_dict):
    current_sample = []
    reduced_filter = []
    for dicts in filter_dict:  # List of dictionaries
        if dicts["Sample"] not in current_sample:
            reduced_filter.append(dicts)
            current_sample.append(dicts["Sample"])
    return reduced_filter

def filter_graphing(filter_dicts, error_dict=None):
    filter_dfs = []
    error_metrics = {}
    if error_dict is not None:
        error_m = {"Otsu":{}, "Yen": {}, "Mean": {}, "Triangle": {}, "Li": {}}
        for err in error_dict:
            error_m[err["Filter Type"]][err["Sample"]] = err
        error_metrics = error_m

    for f in list(filter_dicts):
        filter_dict = filter_dicts[f]
        reduced_filter = filter_reduce(filter_dict)
        reduced = []
        for red in reduced_filter:
            red["Richard"] = manual_Hysteresis[red['Sample']][0][0] * 255
            red["Rensu"] = manual_Hysteresis[red['Sample']][1][0] * 255
            red["Average"] = (red["Richard"] + red["Rensu"]) / 2
            red["Diff1"] = abs(red["Richard"] - red["Low_Thresh"]) / red["Richard"]
            red["Diff2"] = abs(red["Rensu"] - red["Low_Thresh"]) / red["Rensu"]
            red["DiffAv"] = abs(red["Average"] - red["Low_Thresh"]) / red["Average"]
            red["Filter Type"] = f
            if error_metrics:
                red["Error1"] = error_metrics[red["Filter Type"]][red["Sample"]]["Richard Error"]
                red["Error2"] = error_metrics[red["Filter Type"]][red["Sample"]]["Rensu Error"]
                red["ErrorAv"] = (red["Error1"] + red["Error2"])/2
            reduced.append(red)
            filter_dfs.append(red)
    filter_df = pd.DataFrame(filter_dfs)
    new_filter = filter_df.set_index("Filter Type")
    column_list = ["Diff1", "Diff2", "DiffAv"]
    if error_metrics:
        column_list = column_list + ["Error1", "Error2", "ErrorAv"]
    new_filter = new_filter[column_list]
    mean_df = new_filter.groupby(level='Filter Type').mean()
    print(new_filter)
    print(mean_df)
    ax1 = mean_df[["Diff1", "Diff2", "DiffAv"]].plot.bar()
    ax2 = mean_df[["Error1", "Error2", "ErrorAv"]].plot.bar()
    plt.show()

def intensity_graphs(intensity_dict_list, filter, dens):
    intensities_by_sample = {}
    for i in intensity_dict_list:
        if i["Sample"] not in intensities_by_sample:
            intensities_by_sample[i["Sample"]] = []
        if i["Filter Type"] == filter and i["Starting Density"] == dens:
            intensities_by_sample[i["Sample"]].append({"Precision": i["Precision"], "Total Voxels": i["Total Voxels"],
                                                       "Voxels per intensity": i["Voxels per intensity"], "Intensity Range": i["Intensity Range"],
                                                       "High_Thesh": i["High_Thesh"]})

    for k, v in intensities_by_sample.items():
        #plt.clf()
        print(k)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle(k)
        gradient_threshes = []
        for j in v:
            print("***************************************************")
            intens_range = j["Intensity Range"][1:]
            voxel_range = j["Voxels per intensity"][:-1]
            print("Total Check:", sum(j["Voxels per intensity"]), j["Total Voxels"], j["Intensity Range"][0], j["Intensity Range"][-1])
            change_in_voxels = []
            perc_voxels_per_intensity = []
            voxel_diff = []
            total_vox = sum(voxel_range)
            vox_gradient = []
            for vox in voxel_range:
                perc_voxels_per_intensity.append(vox)
            perc_voxels_per_intensity.reverse()
            print("Checking Intensity Range:", intens_range)
            print("Checking Voxel Range:", voxel_range)
            for vox in range(1, len(voxel_range)):
                vox_gradient.append((voxel_range[vox] - voxel_range[vox - 1])/(intens_range[vox] - intens_range[vox - 1])*-1)
                change_in_voxels.append((voxel_range[vox]/voxel_range[vox - 1]) - 1)
                voxel_diff.append(voxel_range[vox] - voxel_range[vox - 1])
            print("Checking Voxel Change:", change_in_voxels)
            vox_gradient.reverse()
            max_gradient = max(vox_gradient)
            max_grad_index = vox_gradient.index(max_gradient)
            vox_max = sum(voxel_diff)
            vox_diff_change = []
            orig_intens_range = []
            for r in intens_range:
                orig_intens_range.append(r)
            orig_intens_range.reverse()
            #orig_intens_range = intens_range
            #orig_intens_range.reverse()
            #perc_voxels_per_intensity.reverse()
            intens_range = intens_range[1:]
            voxel_diff.reverse()
            average_of_change = sum(change_in_voxels) / (len(intens_range) - 1)
            average_of_change2 = sum(voxel_diff) / (len(voxel_diff))
            intens_range.reverse()
            for vox in range(1, len(voxel_diff)):
                print(intens_range[vox], intens_range[vox - 1], voxel_diff[vox], voxel_diff[vox - 1], voxel_diff[vox]/voxel_diff[vox - 1])
                vox_diff_change.append(voxel_diff[vox]/voxel_diff[vox - 1])
            #vox_diff_change.reverse()
            gradient_threshes.append(intens_range[max_grad_index])
            print("Max Gradient", max_gradient, "Max Index", max_grad_index, intens_range[max_grad_index])
            change_in_voxels.reverse()
            potential_high_thresh = intens_range[max_grad_index]
            high_thresh_voxels = intens_range.index(j["High_Thesh"])
            orig_thresh_voxels = orig_intens_range.index(j["High_Thesh"])
            voxel_range.reverse()
            print("Voxels per intensity:", voxel_range)
            print("Voxel Differences:", voxel_diff)
            print("Vox Gradient:", vox_gradient)
            print(intens_range, "\n", intens_range[high_thresh_voxels])
            print("-----------------------------")
            print(intens_range, "\n", change_in_voxels)
            print("Change at High:", change_in_voxels[high_thresh_voxels])
            change_around = "Change around: "
            intensity_around = "Intensities around: "
            if high_thresh_voxels > 0:
                change_around += str(change_in_voxels[high_thresh_voxels - 1]) + " "
                intensity_around += str(intens_range[high_thresh_voxels - 1]) + " "
            change_around += str(change_in_voxels[high_thresh_voxels]) + " "
            intensity_around += str(intens_range[high_thresh_voxels]) + " "
            if high_thresh_voxels < len(change_in_voxels) - 1:
                change_around += str(change_in_voxels[high_thresh_voxels + 1]) + " "
                intensity_around += str(intens_range[high_thresh_voxels + 1]) + " "
            print(change_around)
            print(intensity_around)
            print("Average1", average_of_change)
            print("Average2", average_of_change2)
            ax1.plot(intens_range, change_in_voxels, '-D', markevery=high_thresh_voxels, label=str(j["Precision"]))
            ax2.plot(vox_gradient, '-', label=str(j["Precision"]))
            ax3.plot(intens_range, voxel_diff, '-D', markevery=high_thresh_voxels, label=str(j["Precision"]))
            print("Precision:", j["Precision"], " High Thresh", j["High_Thesh"], max_grad_index, intens_range[max_grad_index])
        print(gradient_threshes)
        ax1.set_title("Relative Change")
        ax2.set_title("Gradient")
        ax3.set_title("Flat Voxel Differences")
        plt.legend()
        plt.show()

def intensity_graphs2(intensity_dict_list, filter, dens):
    intensities_by_sample = {}
    for i in intensity_dict_list:
        if i["Sample"] not in intensities_by_sample:
            intensities_by_sample[i["Sample"]] = []
        if i["Filter Type"] == filter and i["Starting Density"] == dens:
            intensities_by_sample[i["Sample"]].append({"Precision": i["Precision"], "Total Voxels": i["Total Voxels"],
                                                       "Voxels per intensity": i["Voxels per intensity"], "Intensity Range": i["Intensity Range"],
                                                       "Voxel Changes": i["Voxel Changes"], "High_Thesh": i["High_Thesh"]})

    for k, v in intensities_by_sample.items():
        #plt.clf()
        print(k)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle(k)
        gradient_threshes = []
        print(v)
        high_thresh_set = {}
        scaled_vox_average_set = {}
        scaled_vox_indexes = {}
        for j in v:
            intens_range = j["Intensity Range"]
            voxel_range = j["Voxels per intensity"]
            changes_range = j["Voxel Changes"]
            total_voxels = j["Total Voxels"]
            measured_total = sum(voxel_range)
            print(j["Precision"])
            print("Measured vs Actual Total:", measured_total, total_voxels)
            print("Provided High Thresh:", j["High_Thesh"])
            print("***************************************************")
            print("Number of intensity values:", len(intens_range))
            print(intens_range, "\n", voxel_range, "\n", changes_range, "\n")
            print("***************************************************")
            max_intensity = 255
            high_thresh_set[str(j["Precision"])] = j["High_Thesh"]
            shorter_range = []
            rescaled_voxel_diffs = []
            for t in range(0, len(intens_range)):
                if t == 0:
                    prior_step = 255
                else:
                    prior_step = intens_range[t - 1]
                current_step = intens_range[t]
                rescaled_voxel_diffs.append(voxel_range[t]/(prior_step - current_step))
            print("Rescaled Vox Differences:", rescaled_voxel_diffs)
            calculated_changes = []
            for s in range(1, len(intens_range)):
                calculated_changes.append((voxel_range[s]/voxel_range[s - 1]))
                shorter_range.append(intens_range[s])
            accumulated_range = []
            calculated_average = sum(calculated_changes[1:])/len(shorter_range[1:])
            total = 0
            average_change = sum(changes_range)/len(shorter_range)
            print("Calculated Voxel Changes:", calculated_changes)
            print(len(calculated_changes) == len(shorter_range))
            print("Average Change:", average_change, "Calculated Average:", calculated_average)
            for c in voxel_range:
                total += c
                accumulated_range.append(total)
            other_change_calc = []
            for a in range(1, len(accumulated_range)):
                other_change_calc.append((accumulated_range[a]/accumulated_range[a - 1]) - 1)
            print("Accumulated Change Difference:", other_change_calc)
            intens_range.reverse()
            scaled_vox_average_index = 1
            scaled_vox_average = sum(rescaled_voxel_diffs) / len(rescaled_voxel_diffs)
            rescaled_voxel_diffs.reverse()
            for resc in range(len(rescaled_voxel_diffs) - 1, 0, -1):
                if rescaled_voxel_diffs[resc] >= scaled_vox_average:
                    scaled_vox_average_index = resc
                    scaled_vox_indexes[str(j["Precision"])] = intens_range[scaled_vox_average_index]
                    print("Rescaled Voxel Index", rescaled_voxel_diffs[resc], resc)
                    break
            print("Intensity Range:", intens_range)
            print("Scaled Index:", scaled_vox_average_index)
            print("Scaled voxel values:", rescaled_voxel_diffs)
            print("Scaled Average:", scaled_vox_average)
            scaled_vox_average_set[str(j["Precision"])] = scaled_vox_average
            voxel_range.reverse()
            changes_range.reverse()
            shorter_range.reverse()
            accumulated_range.reverse()
            high_thresh_voxels = intens_range.index(j["High_Thesh"]) - 1
            print("High Thresh Index:", high_thresh_voxels)
            ax1.plot(intens_range, voxel_range, '-D', markevery=high_thresh_voxels, label=str(j["Precision"]))
            ax2.plot(intens_range, rescaled_voxel_diffs, '-D', markevery=scaled_vox_average_index, label=str(j["Precision"]))
            ax3.plot(intens_range, rescaled_voxel_diffs, '-D', label=str(j["Precision"]))
        print("High Thresholds:", high_thresh_set)
        print("Average Rescaled Voxels:", scaled_vox_average_set)
        print("Average Rescaled Intensities:", scaled_vox_indexes)
        print("Average Rescaled Intensity:", sum(list(scaled_vox_indexes.values()))/len(list(scaled_vox_indexes)))
        set_of_values = set(list(high_thresh_set.values()))
        print("Average High Thresh:", sum(list(set_of_values))/len(list(set_of_values)))
        ax1.set_title("Voxels per intensity")
        ax2.set_title("Rescaled Voxels per intensity")
        ax3.set_title("Rescaled Voxels per intensity every point")
        plt.legend()
        plt.show()

def convert_to_float(list_of_strings):
    if list_of_strings is not None:
        holder = []
        for c in list_of_strings:
            holder.append(float(c))
        return holder
    else:
        return None

def intensity_graphs3(intensity_dict_list, filter, dens, blocked_prec = []):
    intensities_by_sample = {}
    #print(intensity_dict_list)
    input_paths = ["C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\"]
    for i in intensity_dict_list:
        if i["Sample"] not in intensities_by_sample:
            intensities_by_sample[i["Sample"]] = []
        if i["Filter Type"] == filter and i["Starting Density"] == dens:
            temp_dict = {}
            intens_range_dict = {}
            print(i["Sample"])
            for c in ["Fixed", "AdjDensity", "FixedDensity"]:
                if i["Intensity Range"][c] is not None:
                    intens_range_dict[c] = i["Intensity Range"][c]
                    for l in ["Looped", "Unlooped"]: #Removed Looped as it was removed to save processing time
                        temp_dict[(l, c)] = {"Scaled Voxels per intensity": convert_to_float(i["Scaled Voxels per intensity"][l + " - " + c]),
                                            "Voxels per intensity": convert_to_float(i["Voxels per intensity"][l + " - " + c]),
                                            "Voxel Changes": convert_to_float(i["Voxel Changes"][l + " - " + c]), "High_Thresh": i["High_Thesh"][l + " - " + c]}
            intensities_by_sample[i["Sample"]].append({"Precision": i["Precision"], "Total Voxels": i["Total Voxels"], "Intensity Range":intens_range_dict,
                                                       "Low_Thresh": i["Low_Thresh"], "Image Max": i["Image Max"], "Varied Outputs": temp_dict})
    per_sample_error = {}
    across_samples_thresholds = {}
    for k, v in intensities_by_sample.items():
        sample_name = k.split("\\")[-1]
        across_samples_thresholds[sample_name] = {}
        print("Sample:", sample_name)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle(sample_name)
        average_scaled_per_prec = {}
        average_calc_scaled = {}
        intensities_per_prec = {}
        scaled_voxels_per_prec = {}
        scaled_compare = {}
        voxel_per_prec = {}
        intensity_to_be_used = {}
        scaled_vox_indexes = {}
        old_scaled_vox_indexes = {}
        prec_scaled_vox_indexes = {}
        low_thresh = {}
        image_paths = source_images(sample_name, input_paths)
        image = io.imread(image_paths[1])
        image_max = 0
        max_intens = {}
        intersects_per_prec = {}
        intersect_ave = {}
        orig_intersect_ace = []
        for j in v:
            prec = j["Precision"]
            if prec not in blocked_prec:
                image_max = j["Image Max"]
                low_thresh[prec] = j["Low_Thresh"]
                total_v = j["Total Voxels"]
                intens_range = j["Intensity Range"]  # Dictionary of three intensity range variations
                # print("Stuff", intens_range)
                intensities_per_prec[prec] = sub_list_detect(intens_range["FixedDensity"])
                max_intens[prec] = max(sub_list_detect(intens_range["AdjDensity"]))
                varied_outputs = {"Unlooped": {}, "Looped": {}}
                intensity_range_configs = ["AdjDensity", "FixedDensity"]  # Removed "Fixed" as it is not in the latest data
                for t in intensity_range_configs:
                    varied_outputs["Unlooped"][t] = j["Varied Outputs"][("Unlooped", t)]
                    varied_outputs["Looped"][t] = j["Varied Outputs"][("Looped", t)]
                # Going to use just unlooped first
                intens_to_be_used = sub_list_detect(intens_range["FixedDensity"])
                high_thresh_orig = varied_outputs["Unlooped"]["FixedDensity"]["High_Thresh"]
                #print("Intensity Range:", intens_to_be_used)
                # print(varied_outputs["Unlooped"]["AdjDensity"])
                voxel_per_prec_val = varied_outputs["Unlooped"]["FixedDensity"]["Voxels per intensity"]
                voxel_per_prec[prec] = voxel_per_prec_val
                rescaled_voxel_diffs = []
                prec_rescaled = []
                for t in range(0, len(intens_to_be_used)):
                    if t == 0:
                        prior_step = image_max
                    else:
                        prior_step = intens_to_be_used[t - 1]
                    current_step = intens_to_be_used[t]
                    #print(prior_step, current_step)
                    if current_step != prior_step:
                        rescaled = (voxel_per_prec_val[t] / (prior_step - current_step))
                    else:
                        rescaled = (voxel_per_prec_val[t])
                    rescaled_voxel_diffs.append(rescaled)
                    prec_rescaled.append(rescaled * (0.5 - prec))
                scaled_vox_average = (sum(rescaled_voxel_diffs) / len(rescaled_voxel_diffs))
                prec_rescaled_average = (sum(prec_rescaled) / len(prec_rescaled))
                average_scaled_per_prec[prec] = scaled_vox_average
                scaled_compare[prec] = rescaled_voxel_diffs
                scaled_vox_average_index = 1
                rescaled_changes = []
                #print(voxel_per_prec_val)
                #print(rescaled_voxel_diffs)
                for rvd in range(1, len(rescaled_voxel_diffs)):
                    top_value = rescaled_voxel_diffs[rvd] if rescaled_voxel_diffs[rvd] != 0 else 0.000000000000000000000000000000000000000000000001
                    bottom_value = rescaled_voxel_diffs[rvd - 1] if rescaled_voxel_diffs[rvd - 1] != 0 else 0.000000000000000000000000000000000000000000000001
                    rescaled_changes.append(top_value / bottom_value)
                # print("Rescaled Voxle Diffs:", rescaled_voxel_diffs)
                # print("Rescaled Changes:", rescaled_changes)
                acc_changes = []
                if image_max != intens_to_be_used[0]:
                    acc_ch = voxel_per_prec_val[0]/(image_max-intens_to_be_used[0])
                else:
                    acc_ch = voxel_per_prec_val[0]
                for vppv_index in range(1, len(voxel_per_prec_val)):
                    old_acc_tot = acc_ch
                    acc_ch += voxel_per_prec_val[vppv_index]/(image_max - intens_to_be_used[vppv_index])
                    acc_changes.append((acc_ch/old_acc_tot) - 1)
                acc_change_ave = sum(acc_changes)/len(acc_changes)
                answer = 0
                answer_index = 0
                for cbi in range(0, len(acc_changes), 1):
                    if acc_changes[cbi] <= acc_change_ave:
                        # print("The best threshold is at", range_of_intensities[cbi+1])
                        answer_index = cbi + 1
                        answer = intens_to_be_used[cbi + 1]
                        break
                intersect_orig = determine_intersect(intens_to_be_used[1:], acc_changes, acc_change_ave)
                #print("Precision:", prec, "With a threshold at", answer)
                #print("Precision:", prec, "With an intersect at", sum(intersect_orig)/len(intersect_orig))
                orig_intersect_ace.append(sum(intersect_orig)/len(intersect_orig))
                rescaled_voxel_diffs.reverse()
                prec_rescaled.reverse()
                rescaled_changes.reverse()
                intens_to_be_used.reverse()
                acc_voxels_per_intensity = []
                running_total = 0
                for acc in voxel_per_prec_val:
                    running_total += acc
                    acc_voxels_per_intensity.append(running_total)
                calculated_changes = []
                for c in range(1, len(acc_voxels_per_intensity), 1):
                    calculated_changes.append((acc_voxels_per_intensity[c] / acc_voxels_per_intensity[c - 1]) - 1)
                calc_change_average = sum(calculated_changes) / len(calculated_changes)

                calculated_changes.reverse()
                voxel_per_prec_val.reverse()
                acc_voxels_per_intensity.reverse()
                intersects = determine_intersect(intens_to_be_used[:-1], calculated_changes, calc_change_average)
                intersects_per_prec[prec] = intersects
                inters_ave = sum(intersects) / len(intersects)
                intersect_ave[prec] = inters_ave

                for resc in range(len(rescaled_voxel_diffs) - 1, 0, -1):
                    if rescaled_voxel_diffs[resc] >= scaled_vox_average:
                        scaled_vox_average_index = resc
                        old_scaled_vox_indexes[str(j["Precision"])] = intens_to_be_used[scaled_vox_average_index]
                        break
                for prec_resc in range(len(prec_rescaled) - 1, 0, -1):
                    if prec_rescaled[prec_resc] >= prec_rescaled_average:
                        scaled_vox_prec_average_index = prec_resc
                        prec_scaled_vox_indexes[str(j["Precision"])] = intens_to_be_used[scaled_vox_prec_average_index]
                        break
                # scaled_vox_average_intensity = intens_to_be_used[scaled_vox_average_index]
                rescaled_intensity_intersects = determine_intersect(intens_to_be_used, rescaled_voxel_diffs, scaled_vox_average)
                scaled_vox_average_intensity = sum(rescaled_intensity_intersects) / len(rescaled_intensity_intersects)
                scaled_vox_indexes[prec] = scaled_vox_average_intensity
                acc_changes.reverse()
                # intensity_to_be_used[prec] = intens_to_be_used[scaled_vox_average_index]
                ax1.plot(intens_to_be_used, acc_voxels_per_intensity, '-D', label=str(j["Precision"]))
                ax2.plot(intens_to_be_used[:-1], acc_changes, '-D', label=str(j["Precision"]))
                ax3.plot(intens_to_be_used[:-1], calculated_changes, '-D', label=str(j["Precision"]))
                colour = plt.gca().lines[-1].get_color()
                ax1.axhline(y=scaled_vox_average, xmin=0, xmax=1, color=colour)
                ax1.axvline(x=scaled_vox_average_intensity, ymin=0, ymax=1, color=colour)
                ax2.axhline(y=acc_change_ave, xmin=0, xmax=1, color=colour)
                ax2.axvline(x=answer, ymin=0, ymax=1, color=colour)
                ax3.axhline(y=calc_change_average, xmin=0, xmax=1, color=colour)
                ax3.axvline(x=inters_ave, ymin=0, ymax=1, color=colour)
        '''print("Intensities per prec", intensities_per_prec)
        print("Voxels per intensity", voxel_per_prec)
        print("Rescaled Voxels", scaled_compare)
        print("Average of scaled", average_scaled_per_prec)'''
        mean_rescaled_intense = sum(list(scaled_vox_indexes.values())) / len(list(scaled_vox_indexes))
        mean_prec_rescaled_intense = sum(list(prec_scaled_vox_indexes.values())) / len(list(prec_scaled_vox_indexes))
        old_mean_rescaled_intense = sum(list(old_scaled_vox_indexes.values())) / len(list(old_scaled_vox_indexes))
        #print("Max intensities per prec", max_intens)
        #print("Image Max", image_max)
        #print("Average Rescaled Intensity:", mean_rescaled_intense)
        #print("Old Average Rescaled Intensity:", old_mean_rescaled_intense)
        #print("Average Rescaled Weighted by Stepsize:", mean_prec_rescaled_intense)
        #print("Richard High", manual_Hysteresis[sample_name][0][1] * 255, "Richard Low", manual_Hysteresis[sample_name][0][0] * 255)
        #print("Rensu High", manual_Hysteresis[sample_name][1][1] * 255, "Rensu Low", manual_Hysteresis[sample_name][1][0] * 255)
        #print("Low Thresh", low_thresh[0.02])
        '''
        print("Scaled Intensity", scaled_vox_indexes)
        print("Scaled Intensities rescaled by prec", prec_scaled_vox_indexes)
        print("Intersection points:", intersects_per_prec)
        print("Intersection Averages:", intersect_ave)'''
        intersect_values = list(intersect_ave.values())
        intersect_high = sum(intersect_values) / len(intersect_values)
        #print("Changes Intensity Value:", intersect_high)
        orig_intersect_high = sum(orig_intersect_ace)/len(orig_intersect_ace)
        across_samples_thresholds[sample_name] = {"Richard":{"Low:":manual_Hysteresis[sample_name][0][0] * 255, "High:":manual_Hysteresis[sample_name][0][1] * 255},
                                                  "Rensu":{"Low:":manual_Hysteresis[sample_name][1][0] * 255, "High:":manual_Hysteresis[sample_name][1][1] * 255},
                                                  "Auto":{"Low:":low_thresh[0.02], "Ave Rescaled:":mean_rescaled_intense, "Old Ave Rescaled:":old_mean_rescaled_intense,
                                                          "Change Intensity:":intersect_high, "Orig High:":high_thresh_orig,
                                                          "Calc Orig Intersect": orig_intersect_high}}
        errors1 = calculate_error_values(sample_name, low_thresh[0.02], mean_rescaled_intense, image, False)
        '''errors2 = calculate_error_values(sample_name, low_thresh[0.02], intersect_high, image, False)
        print("Errors with Rescaled:", errors1)
        print("Errors with Intersect:", errors2)
        per_sample_error[sample_name] = {"Rescale Errors:": errors1, "Intersect Errors": errors2}'''
        per_sample_error[sample_name] = errors1
        ax1.set_title("Total Voxels per intensity")
        ax2.set_title("Voxels per intensity")
        ax3.set_title("Rescaled Voxel Changes")
        ax1.axvline(x=mean_rescaled_intense, ymin=0, ymax=1, color='k')
        ax3.axvline(x=intersect_high, ymin=0, ymax=1, color='k')
        plt.legend()
        #plt.show()
        for k, v in low_thresh.items():
            low = v
            break
        save_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Data Results 3\\Ave_resc\\"
        generate_composite_img(sample_name, image, low, mean_rescaled_intense, save_path)
    for key, value in across_samples_thresholds.items():
        print("Sample", key)
        for k, v in value.items():
            print(k, v)
    print(across_samples_thresholds)
    print("Errors across all samples", per_sample_error)


def flatten_list(nested_iter):
    new_list = []
    try:
        current_value = next(nested_iter)
        if type(current_value) is list:
            sub_iter = iter(current_value)
            new_list += flatten_list(sub_iter)
            resumed_result = flatten_list(nested_iter)
            if resumed_result is not None:
                new_list += resumed_result
        else:
            new_list += [current_value]
            next_value = flatten_list(nested_iter)
            if next_value is None:
                return new_list
            else:
                return new_list + next_value
    except StopIteration:
        return None
    return new_list



def sub_list_detect(nested_lists):
    found = False
    nested_list = None
    for n in nested_lists:
        if type(n) is list:
            nested_list = sub_list_detect(n)
            found = True
    if found:
        nested_lists = nested_list
    return nested_lists



def calculate_error_values(sample_name, low_thresh, high_thresh, image, render=True):
    richard_thresh_params = manual_Hysteresis[sample_name][0]
    rensu_thresh_params = manual_Hysteresis[sample_name][1]
    image_max = np.max(image)
    richard_image = apply_hysteresis_threshold(image, richard_thresh_params[0]*image_max, richard_thresh_params[1]*image_max) * 255
    rensu_image = apply_hysteresis_threshold(image, rensu_thresh_params[0] * image_max, rensu_thresh_params[1] * image_max) * 255
    thresholded_images = [richard_image, rensu_image]
    average_image = np.mean(np.array(thresholded_images), axis=0)
    auto_image = apply_hysteresis_threshold(image, low_thresh, high_thresh) * 255
    richard_error = np.average(np.abs(richard_image - auto_image))
    rensu_error = np.average(np.abs(rensu_image - auto_image))
    average_error = np.average(np.abs(average_image - auto_image))
    if render:
        overlayed_img = np.stack((richard_image, rensu_image, auto_image), axis=-1)
        render_image_sequence(overlayed_img, sample_name + " Composite Image")
        #render_image_sequence(richard_image, sample_name + " Richard Image")
        #render_image_sequence(rensu_image, sample_name + " Rensu Image")
        render_image_sequence(auto_image, sample_name + " Auto Image")
        plt.show()
    return {"Richard":richard_error, "Rensu":rensu_error, "Average":average_error}

def generate_composite_img(sample_name, image, low, high, save_path):
    richard_thresh_params = manual_Hysteresis[sample_name][0]
    rensu_thresh_params = manual_Hysteresis[sample_name][1]
    image_max = np.max(image)
    richard_image = (apply_hysteresis_threshold(image, richard_thresh_params[0]*255, richard_thresh_params[1]*255) * 255).astype('uint8')
    rensu_image = (apply_hysteresis_threshold(image, rensu_thresh_params[0] * 255, rensu_thresh_params[1] * 255) * 255).astype('uint8')
    thresholded_images = [richard_image, rensu_image]
    #average_image = np.mean(np.array(thresholded_images), axis=0)
    print("Low", low, "High", high)
    auto_image = (apply_hysteresis_threshold(image, low, high) * 255).astype('uint8')
    overlayed_img = np.stack((richard_image, rensu_image, auto_image), axis=-1)
    if not exists(save_path):
        makedirs(save_path)
    io.imsave(save_path + "Overlay" + sample_name, overlayed_img)


def render_image_sequence(image, image_type):
    number_of_slices = image.shape[0]
    columns = int(number_of_slices/2)
    if columns == 0:
        rows = 1
        columns = number_of_slices
    else:
        rows = 2

    fig, axarr = plt.subplots(rows, columns, figsize=(10, 5))
    fig.suptitle(image_type)
    fig.tight_layout()
    slice = 0

    if rows > 1:
        for row in range(rows):
            for column in range(columns):
                axarr[row, column].imshow(image[slice, :, :])
                axarr[row, column].set_title("Slice " + str(slice + 1))
                slice += 1
    else:
        for column in range(columns):
            axarr[column].imshow(image[slice, :, :])
            axarr[column].set_title("Slice " + str(slice + 1))
            slice += 1
    #plt.show()

def determine_intersect(x, y, average):
    intersection_points = []
    for i in range(0, len(y)):
        if y[i] == average:
            intersection_points.append(x[i])
        else:
            if i > 0 and ((y[i - 1] < average < y[i]) or (y[i - 1] > average > y[i])):
                intens_delta = (y[i] - average)*((x[i] - x[i - 1])/(y[i] - y[i - 1]))
                inters = x[i] - intens_delta
                intersection_points.append(inters)
    return intersection_points

def source_images(sample_name, input_paths):
    files = [[f, input_path + f] for input_path in input_paths for f in listdir(input_path) if isfile(join(input_path, f))]
    for s in files:
        #print(s[0].split('Noise')[0] + '.tif')
        if sample_name in s[0].split('Noise')[0] + '.tif':
            return [sample_name, s[1]]

def thresholded_image_view(img_path, sample_name, low_thresh, high_thresh):
    img = io.imread(img_path + sample_name.split(".")[0] + "Noise000" + ".tif")
    thresholded_image = apply_hysteresis_threshold(img, low_thresh, high_thresh)*255
    thresholded_image2 = apply_hysteresis_threshold(img, 25.5, 104.04) * 255
    for slice in range(thresholded_image.shape[0]):
        print("Slice:", slice)
        io.imshow(thresholded_image[slice])
        plt.show()
        plt.clf()
        io.imshow(thresholded_image2[slice])
        plt.show()

def image_stitcher(img_paths):
    files = [[f, input_path] for input_path in img_paths for f in listdir(input_path) if isfile(join(input_path, f))]
    image_list = []
    json_list = []
    organized_images = {}
    for file in files:
        if file[0].endswith(".json") and ".tif" in file[0]:
            json_list.append(file[0].split(".tif")[0])
        elif file[0].endswith(".tif") and "StackedSteps" not in file[0]:
            image_list.append(file)
    for i in image_list:
        #print(i)
        for j in json_list:
            if j in i[0] and j:
                if j not in organized_images:
                    organized_images[j] = {}
                    organized_images[j]["Path"] = i[1]
                params = i[0].split(j)[0].split("StepSize")
                stepSize = float(params[1])
                filter_type = params[0].split("Filter")[1]
                if filter_type not in organized_images[j]:
                    organized_images[j][filter_type] = []
                organized_images[j][filter_type].append(stepSize)
                organized_images[j][filter_type].sort()
    for sample, details in organized_images.items():
        path = details['Path']
        mean_filter = details['Mean']
        complete_image_stack = []
        print("Sample:", sample)
        for m in mean_filter:
            file_name = "FilterMean" + "StepSize" + str(m) + sample + ".tif"
            stored_image = io.imread(path + file_name)
            print("Step Size:", m, "Max:", stored_image.max())
            complete_image_stack.append(stored_image)
        stacked_array = np.stack(complete_image_stack)
        #print(sample, stacked_array.shape)
        #tifffile.imwrite(path + "StackedSteps" + sample + ".tif", stacked_array, imagej=True, metadata={'axes': 'TZYX'})


def generate_histogram():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisPreview\\"
    histogram_data = "HysteresisPreviewGraphs.json"
    f = open(input_path + histogram_data)
    data = json.load(f)
    f.close()
    for k, v in data.items():
        manual_params = manual_Hysteresis[k]
        richard_high = manual_params[0][1]*255
        rensu_high = manual_params[1][1]*255
        plt.axvline(x=richard_high, color='r')
        plt.axvline(x=rensu_high, color='b')
        intens = v[0]
        voxels = v[1]
        plt.xlabel("Intensity Value")
        plt.ylabel("Threshold Voxels")
        plt.title("Threshold Voxels per High Threshold Intensity for " + k)
        plt.plot(intens, voxels)
        plt.savefig(input_path + "HysteresisHist" + k.split(".")[0] + ".png")
        plt.clf()

def cum_av_intersections(x, y, smoothed):
    y_max = max(y)
    if len(x) != len(y) or len(y) != len(smoothed):
        print("Not matching", len(x), len(y), len(smoothed))
        return []
    else:
        list_of_intersection_points = []
        for d in range(0, len(x)):
            if smoothed[d] == y[d]:
                list_of_intersection_points.append(x[d])
            else:
                if d > 0 and ((y[d - 1] < smoothed[d] < y[d]) or (y[d - 1] > smoothed[d] > y[d])):
                    intens_delta = (y[d] - smoothed[d]) * ((x[d] - x[d - 1]) / (y[d] - y[d - 1]))
                    inters = x[d] - intens_delta
                    list_of_intersection_points.append(inters)
        mean_of_inter = sum(list_of_intersection_points)/len(list_of_intersection_points)
        return list_of_intersection_points, mean_of_inter


def gaussian_weights(values, sensitivity=1):
    max_val = math.ceil(max(values))
    resized = 1
    if max_val == 1:
        smallest_gap = 1
        for v in range(1, len(values)):
            gap = abs(values[v-1] - values[v])
            if gap < smallest_gap and gap != 0:
                smallest_gap = gap
        np_values = np.array(values)
        max_val = math.ceil((np_values/smallest_gap).max())
        resized = smallest_gap
    width = max_val/0.997
    sigma = (width/3)*sensitivity
    mu = 0
    distr = np.random.normal(mu, sigma, 2*int(max_val))
    if max_val > 1:
        x_axis = np.linspace(0, int(max_val)+1, int(max_val)+1)
    else:
        x_axis = np.linspace(0, 10, 10)
    bins = x_axis
    dens = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2))
    dens = dens/np.max(dens)
    dens_index = np.argmax(dens)
    dens = dens[dens_index:]
    rescaled_values = []
    intensity_scaled = []
    for dv in range(len(values)):
        vald = values[dv] / resized
        rescaled_values.append(dens[int(vald)])
        intensity_scaled.append(vald * dens[int(vald)])
    '''plt.plot(np.linspace(0, len(rescaled_values), len(rescaled_values)), rescaled_values)
    plt.show()'''

    return rescaled_values

def rescale_gaussian(values, sensitivity=0, voxels=None, power=1):
    max_val = math.ceil(max(values))
    resized = 1
    voxel_weights = []
    calculated_centroid = centroid(values)
    '''plt.plot(np.linspace(0, len(values), len(values)), values)
    plt.title("Centroid of Slope Graph")
    plt.axvline(x=calculated_centroid, color='k')
    plt.show()'''
    if voxels is not None:
        #This will be for a weighting to determine the percentage of structures thresholded at a specific high threshold
        voxel_array = (np.array(voxels, dtype="int64")**power).tolist()
        maximum_voxels = max(voxel_array)
        for voxel in voxel_array[:-1]:
            voxel_weights.append(voxel/maximum_voxels)
        print(voxel_weights)
    if max_val == 1:
        smallest_gap = 1
        for v in range(1, len(values)):
            gap = abs(values[v-1] - values[v])
            if gap < smallest_gap and gap != 0:
                smallest_gap = gap
        np_values = np.array(values)
        max_val = math.ceil((np_values/smallest_gap).max())
        resized = smallest_gap
    width = max_val/0.997
    sigma = width/(3 + sensitivity)
    #print("Sigma", sigma)
    mu = 0
    distr = np.random.normal(mu, sigma, 2*int(max_val))
    if max_val > 1:
        x_axis = np.linspace(0, int(max_val)+1, int(max_val)+1)
    else:
        x_axis = np.linspace(0, 10, 10)
    bins = x_axis
    dens = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2))
    dens = dens/np.max(dens)
    dens_index = np.argmax(dens)
    dens = dens[dens_index:]
    #plt.plot(bins, dens)
    print("Min for weighting", min(values), min(values)/resized)
    range_of_means1 = 0
    range_of_means2 = 0
    range_weights = 0
    rescaled_values = []
    means = []
    intensity_scaled = []
    intensity_weights = []
    rescaled_x = []
    for dv in range(len(values)):
        vald = values[dv] / resized
        rescaled_values.append(dens[int(vald)])
        intensity_scaled.append(vald * dens[int(vald)])
        intensity_weights.append(dens[int(vald)])
        rescaled_x.append(dv * dens[int(vald)])
    inner_knee = KneeLocator(np.linspace(0, len(rescaled_values), len(rescaled_values)), rescaled_values, S=0.1, curve="convex", direction="decreasing")
    '''plt.plot(np.linspace(0, len(rescaled_values), len(rescaled_values)), rescaled_values)
    plt.show()'''
    print("Inner Knee:", inner_knee.all_knees)
    knee = inner_knee.knee
    print("Rescaled at knee", rescaled_values[int(knee)])
    rescaled_values = rescaled_values[int(knee):]
    values = values[int(knee):]
    '''rescaled_centroid1 = centroid(rescaled_values)
    rescaled_centroid2 = centroid(values, weights=rescaled_values)
    print("Rescaled Centroids:", rescaled_centroid1, rescaled_centroid2)
    plt.plot(np.linspace(0, len(rescaled_values), len(rescaled_values)), rescaled_values, color='r')
    plt.title("Rescaled Graph with Centroid")
    plt.axvline(x=rescaled_centroid1, color='k')
    plt.axvline(x=rescaled_centroid2, color='g')
    plt.show()'''
    intensity_scaled = intensity_scaled[int(knee):]
    intensity_weights = intensity_weights[int(knee):]
    auc_value = metrics.auc(np.linspace(0, len(intensity_scaled), len(intensity_scaled)), intensity_scaled)
    print("Auc Value", auc_value)
    max_intens = max(intensity_scaled)
    '''intensity_weights = []
    intensity_num = 0
    intensity_denom = 0
    for iw in range(len(intensity_scaled)):
        intensity_weights.append(intensity_scaled[iw]/max_intens)
        intensity_num += (intensity_scaled[iw]/max_intens)*iw
        intensity_denom += (intensity_scaled[iw]/max_intens)
    intensity_scaled_centroid = intensity_num/intensity_denom'''
    rounded_values = []
    for t in intensity_scaled:
        rounded_values.append(t)
    print("Proper rescaled value range", set(rounded_values), len(list(set(rounded_values))), len(rounded_values))
    print(values, "\n", "intensity_scaled", "\n", intensity_weights, "\n", resized)
    intensity_centroid = sum(intensity_scaled)/sum(intensity_weights)
    rounded_centroid = centroid(rounded_values)
    '''plt.plot(np.linspace(0, len(rounded_values), len(rounded_values)), rounded_values, color='r')
    plt.title("Centroid of Rounded Values")
    plt.axvline(x=rounded_centroid, color='k')
    plt.show()'''
    dft = 0
    denom = 0
    for c in range(len(values)):
        if voxels is not None:
            vox_weight = voxel_weights[c]
        else:
            vox_weight = 1
        dft += c * dens[int(values[c])] * vox_weight
        denom += dens[int(values[c])]
    print("Centroid at", dft/denom + int(knee))
    window_weighted = []
    window_positions = []
    y_max = max(rescaled_values)
    for d in range(len(values), 0, -1):
        sum_scaled = 0
        weights = 0
        flipped = 0
        for v in range(0, d):
            val = values[v]/resized
            weighted_val = v * dens[int(val)]
            weight = dens[int(val)]
            if voxels is not None:
                weighted_val *= voxel_weights[v]
                weight *= voxel_weights[v]
            sum_scaled += weighted_val
            weights += weight
            flipped += (len(values) - v) * dens[int(val)]
        new_mean = (sum_scaled/d)
        #new_mean = new_mean*(d/len(values))
        window_weighted.append(sum_scaled/weights)
        weighted_mean = sum_scaled/weights
        #print("Compared:", weighted_mean, d, weights, new_mean)
        range_of_means1 += new_mean
        means.append(new_mean)
        range_weights += (d + 1)/len(values)
        range_of_means2 += weighted_mean
        #flipped_mean = (len(values) - flipped/len(values))
        '''if int(len(values)*0.25) == d or int(len(values)*0.50) == d or int(len(values)*0.75) == d or int(len(values)) == d:
            window_positions.append(d)
            plt.plot(np.linspace(0, len(rescaled_values), len(rescaled_values)), rescaled_values, label="Rescaled", color='r')
            plt.axvline(x=d, ymin=0, ymax=y_max, color='b')
            plt.axvline(x=weighted_mean+int(knee), color='c')
            plt.show()'''
    print("Window Means:", sum(window_weighted), range_of_means1)
    window_centroid = sum(window_weighted)/len(window_weighted) + int(knee)
    print("Window Centroid", window_centroid)
    other_m = sum(means)/len(means)
    #print("Other M", other_m)
    #print("Current Mean", range_of_means1/len(values))
    print("Length check", len(intensity_scaled), len(values))
    '''for m in means:
        plt.axvline(x=m, color='b', linestyle='--')'''
    '''plt.plot(np.linspace(0, len(intensity_scaled), len(intensity_scaled)), intensity_scaled, label="Rescaled", color='r')
    # plt.plot(np.linspace(0, len(values), len(values)), values, label="Original", color='g')
    plt.axvline(x=range_of_means1/len(rescaled_values), color='k')
    #plt.axvline(x=inner_knee.knee, color='g')
    plt.axvline(x=intensity_scaled_centroid, color='c')'''
    '''for win in window_weighted:
        plt.axvline(x=win, color='c')'''
    '''plt.axvline(x=window_centroid, color='g', label="Centroid")
    plt.legend()
    plt.show()'''
    complete_mean1 = range_of_means1/range_weights
    complete_mean2 = range_of_means2/range_weights
    #print(range_weights, len(values))
    #print("Both final weights", range_of_means1/range_weights, range_of_means2/range_weights, range_of_means1/len(values), range_of_means2/len(values))
    return (range_of_means1/len(rescaled_values)) + int(knee), (range_of_means2/len(rescaled_values)) + int(knee)

def centroid_of_graph(values):
    x_values = np.linspace(0, len(values), len(values))
    area = 0
    for v in values:
        area += v
    centroid = 0
    for i in range(0, len(values)-1):
        y_weight = i*values[i+1] - (i+1)*values[i]
        x_weight = (2*i + 1)
        centroid += x_weight*y_weight

def centroid(y, x=None, weights=None):
    if weights is None:
        max_y = max(y)
        weighting = []
        for w in y:
            weighting.append(w/max_y)
    else:
        weighting = weights

    if x is None:
        x_axis = np.linspace(0, len(y), len(y))
    else:
        x_axis = x
    dtf = []
    denom = []
    for n in range(len(weighting)):
        dtf.append(x_axis[n] * weighting[n])
        denom.append(weighting[n])
    centr = sum(dtf)/sum(denom)
    print("Calculated Centroid", centr)
    return centr

def evaluate_cumulative_hys():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisPreview\\"
    image_paths = ["C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\"]
    files = {(f.split('\\')[-1].split("Noise")[0] + ".tif"): (image_path + f) for image_path in image_paths for f in listdir(image_path) if isfile(join(image_path, f))}
    print(files)
    histogram_data = "HysteresisPreviewGraphs.json"
    f = open(input_path + histogram_data)
    data = json.load(f)
    f.close()
    files_number = len(files)
    file_counter = 0
    for k, v in data.items():
        file_counter += 1
        img_path = files[k]
        print("Sample:", k)
        manual_params = manual_Hysteresis[k]
        richard_high = manual_params[0][1]*255
        rensu_high = manual_params[1][1]*255
        intens = v[0][:-3]
        voxels = v[1][:-3]
        discrete_voxels = []
        for t in range(0, len(voxels)-1):
            discrete_voxels.append(voxels[t] - voxels[t+1])
        discrete_voxels.append(voxels[-1])
        log_voxels = np.where(voxels != 0, np.log2(voxels), 0)
        slopes, slope_points = get_slope(intens, voxels)
        slopes2, slope_points2 = get_slope(intens, log_voxels)
        window_size = 8
        print("Window Size of", window_size)
        mving_slopes = rolling_average(slopes, window_size)
        rolling_centroid = centroid(mving_slopes, slope_points)
        knee_finder = KneeLocator(slope_points, mving_slopes)
        print("Knees", knee_finder.all_knees)
        mving_slopes_intersect, mean_inters = cum_av_intersections(slope_points, slopes, mving_slopes)
        mving_slopes2 = rolling_average(slopes2, window_size)
        '''draw_lines(10, mving_slopes2, slope_points2)
        draw_lines_density(20, mving_slopes2, slope_points2)'''
        #tunable_method(mving_slopes, mving_slopes2)
        rolling_centroid2 = centroid(mving_slopes2, slope_points2)
        mving_slopes_intersect2, mean_inters2 = cum_av_intersections(slope_points2, slopes2, mving_slopes2)
        print("Slope Mean", k)
        voxel_weights = gaussian_weights(mving_slopes2, 2)
        voxel_centroid = centroid(voxels[:-1], intens, voxel_weights)
        square_voxels = []
        for sv in voxels:
            square_voxels.append(sv*sv)
        auto_low = []
        auto_high = []
        set_of_means = []
        for p in [0, 1, 2]:
            mean1, mean2 = rescale_gaussian(mving_slopes, 0, voxels, power=p)
            print("Compared Means:", mean1, mean2)
            mean1 = mean1 + intens[0]
            mean2 = mean2 + intens[0]
            set_of_means.append(mean1)
            auto_low.append(intens[0])
            auto_high.append(mean1)
            print("Sample:", k)
            #to_mip(img_path, intens[0], mean1, manual_params, k)
            print("Mean of", mean1)
            locator = KneeLocator(x=intens, y=voxels, curve="convex", direction="decreasing")
            locatorlog = KneeLocator(x=intens, y=log_voxels, curve="convex", direction="decreasing")
            print(locator.all_knees)
            print(locatorlog.all_knees)
            fig, (ax1, ax3) = plt.subplots(2, layout='constrained')
            #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, layout='constrained')
            fig.suptitle("Threshold Voxels per High Threshold Intensity for " + k + " with voxel power " + str(p))
            plt.xlabel("Intensity Value")
            #plt.ylabel("Threshold Voxels")
            ax1.set_title("Voxels")
            ax1.axvline(x=richard_high, color='r', label="Richard Manual")
            ax1.axvline(x=rensu_high, color='b', label="Rensu Manual")
            #ax1.axvline(x=voxel_centroid, color='m')
            ax1.plot(intens, voxels)
            ax1.axvline(x=mean1, color='k')
            ax1.axvline(x=mean2, color='g')
            '''ax2.set_title("Log of Voxels")
            ax2.axvline(x=richard_high, color='r')
            ax2.axvline(x=rensu_high, color='b')
            ax2.plot(intens, log_voxels)'''
            ax3.set_title("Slope Graph")
            ax3.plot(slope_points, slopes)
            ax3.plot(slope_points, mving_slopes)
            ax3.axvline(x=richard_high, color='r')
            ax3.axvline(x=rensu_high, color='b')
            #ax3.axvline(x=rolling_centroid, color='m')
            ax3.axvline(x=mean1, color='k')
            ax3.axvline(x=mean2, color='g')
            '''ax4.set_title("Voxel Log Slope Graph")
            ax4.plot(slope_points2, slopes2)
            ax4.plot(slope_points2, mving_slopes2)
            ax4.axvline(x=richard_high, color='r')
            ax4.axvline(x=rensu_high, color='b')'''
            #ax4.axvline(x=rolling_centroid2, color='m')
            '''for m in mving_slopes_intersect2:
                ax4.axvline(x=m, color='g')'''
            '''ax4.axvline(x=mean1, color='k')'''
            fig.legend()
        plt.show()
        fig2, (ax2, ax4) = plt.subplots(2, layout='constrained')
        fig2.suptitle("Threshold Voxels per High Threshold Intensity for " + k)
        plt.xlabel("Intensity Value")
        ax2.set_title("Voxels")
        ax2.axvline(x=set_of_means[0], color='r', label="No Voxel Weighting")
        ax2.axvline(x=set_of_means[1], color='g', label="Voxel Weighting")
        ax2.axvline(x=set_of_means[2], color='b', label="Squared Voxel Weighting")
        ax2.plot(intens, voxels, color='k')
        ax4.set_title("Slope Graph")
        ax4.plot(slope_points, slopes, color='k')
        ax4.plot(slope_points, mving_slopes, color='m')
        ax4.axvline(x=set_of_means[0], color='r')
        ax4.axvline(x=set_of_means[1], color='g')
        ax4.axvline(x=set_of_means[2], color='b')
        fig2.legend()
        plt.show()
    print(str(file_counter) + " of " + str(files_number) + " files process")

def draw_lines(window_size, waveform, waveform_x):
    list_of_lines = []
    print("Drawing Lines")
    fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
    half_window = int(window_size / 2)
    for i in range(half_window, len(waveform) - half_window, 1):
        start_x = waveform_x[i - half_window]
        end_x = waveform_x[i + half_window]
        start_y = waveform[i - half_window]
        end_y = waveform[i + half_window]
        list_of_lines.append([[start_x,  end_x], [start_y, end_y]])
    print("List of Lines:", list_of_lines)
    intersection_points = line_iteration(list_of_lines)
    print(intersection_points)
    ax1.plot(waveform_x, waveform, color='k')
    for n in list_of_lines:
        #print(n[0], n[1])
        ax1.plot(n[0], n[1], color='g', linewidth=1)
    ax1.scatter(intersection_points[0], intersection_points[1], s=4, c='r')
    ax2.plot(waveform_x, waveform, color='k')
    ax2.scatter(intersection_points[0], intersection_points[1], s=6, c='r')
    plt.show()

def draw_lines_density(window_size, waveform, waveform_x):
    print("Drawing Lines")
    fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
    intersection_points = [[], []]
    all_lines = []
    for w in range(2, window_size+1):
        list_of_lines = []
        half_window = int(w/2)
        max_values = []
        for i in range(half_window, len(waveform)-half_window, 1):
            upper = (i + half_window)
            lower = (i-half_window)
            max_val = max(waveform[(i-half_window):(i + half_window)])
            if len(max_values) == 0:
                max_values.append(max_val)
            else:
                if max_val > max(max_values):
                    max_values.append(max_val)
            start_x = waveform_x[i-half_window]
            end_x = waveform_x[i + half_window]
            start_y = waveform[i-half_window]
            end_y = waveform[i + half_window]
            list_of_lines.append([[start_x,  end_x], [start_y, end_y]])
        #print("List of Lines:", list_of_lines)
        intersection_point = line_iteration(list_of_lines)
        print("Number of intersections", len(intersection_point[0]))
        intersection_points[0].extend(intersection_point[0])
        intersection_points[1].extend(intersection_point[1])
        all_lines.extend(list_of_lines)
    #print("Intersection Points:" + "\n" + str(intersection_points[0]) + "\n" + str(intersection_points[1]))
    print("Total Number of Intersections", len(intersection_points[0]))
    for n in all_lines:
        # print(n[0], n[1])
        ax1.plot(n[0], n[1], color='g', linewidth=1)
    ax1.plot(waveform_x, waveform, color='k')
    ax1.scatter(intersection_points[0], intersection_points[1], s=4, c='r')
    ax2.plot(waveform_x, waveform, color='k')
    ax2.scatter(intersection_points[0], intersection_points[1], s=6, c='r')
    plt.show()

def line_iteration(lines):
    intersection_points = [[], []]
    for n in range(1, len(lines)):
        current_coords = lines[n]
        for m in range(0, n):
            prior_coords = lines[m]
            if prior_coords[0][1] > current_coords[0][0]:
                #print("Lines", prior_coords, current_coords)
                intersection_found = line_intersection_check(current_coords, prior_coords)
                if intersection_found:
                    intersection = line_intersect(current_coords, prior_coords)
                    if intersection is not None:
                        intersection_points[0].append(intersection[0])
                        intersection_points[1].append(intersection[1])
    return intersection_points

def line_intersection_check(coord1, coord2):
    condition1 = (coord1[1][0] < coord2[1][0] and coord1[1][1] > coord2[1][1])
    condition2 = (coord1[1][0] > coord2[1][0] and coord1[1][1] < coord2[1][1])
    if condition1 or condition2:
        return True
    else:
        return False

def line_intersect(coord1, coord2):
    #coord1 is current, coord2 is prior
    m1 = (coord1[1][1]-coord1[1][0])/(coord1[0][1]-coord1[0][0])
    c1 = coord1[1][0]-m1*coord1[0][0] if m1 < 0 else coord1[1][1]-m1*coord1[0][1] # negative slope needs a y offset
    m2 = (coord2[1][1] - coord2[1][0]) / (coord2[0][1] - coord2[0][0])
    c2 = coord2[1][0]-m2*coord2[0][0] if m2 < 0 else coord2[1][1]-m2*coord2[0][1]
    if m1*m2 < 0:
        x_p = (c1 - c2)/(m2 - m1)
        y_p = (m2*x_p + c2)
        #bound_conditions1 = coord1[0][0] <= x_p <= coord2[0][1]
        bound_conditions2 = (coord1[1][0] <= y_p <= coord1[1][1]) or (coord1[1][1] <= y_p <= coord1[1][0])
        bound_conditions4 = (coord2[1][0] <= y_p <= coord2[1][1]) or (coord2[1][1] <= y_p <= coord2[1][0])
        print("Boundaries")
        print( bound_conditions2, bound_conditions4)
        print([x_p, y_p], coord1, coord2)
        if  bound_conditions2 and bound_conditions4:
            return [x_p, y_p]
        else:
            return None
        #return [x_p, y_p]
    else:
        return None

def rolling_average(counts, window_size=10, rescale=False):
    adjusted = False
    if type(counts) is list and window_size > 1:
        new_list = []
        for n in range(0, int(window_size/2)):
            new_list.append(0)
        counts = new_list + counts + new_list
        adjusted = True
    df = pd.DataFrame(counts)
    moving_average = df.rolling(window_size, center=True).mean()
    average_results = flatten_list(iter(moving_average.values.tolist()))
    if adjusted:
        window_offset = int(window_size/2)
        average_results = average_results[window_offset:-window_offset]
        if rescale:
            print("Prior to rescaling", average_results)
            for i in range(1, window_offset+1):
                average_results[i-1] = (average_results[i-1]*10)/i
                average_results[-i] = (average_results[-i]*10)/i
            print("Rescaled results", average_results)
    return average_results

def get_slope(x, y):
    if len(x) != len(y):
        print("Inconsistent x and y coordinates")
        return None, None
    else:
        slope_values = []
        for i in range(1, len(x), 1):
            slope = abs((y[i] - y[i-1])/(x[i] - x[i-1]))
            slope_values.append(slope)
        #new_x = np.linspace(0, len(slope_values), len(slope_values))
        new_x = x[1:]
        return slope_values, new_x

def biased_mean_test():
    results = []
    x_axis = []
    for i in range(200):
        power = i/10
        x_axis.append(i)
        results.append(math.exp(power))
    total_mean = 0
    list_of_means = []
    max_res = max(results)
    weights = []
    unshifted_mean = 0
    for w in results:
        weights.append(w/max_res)
        unshifted_mean += w
    dft = 0
    for c in range(len(results)):
        dft += c*weights[c]
    print("Centroid?", dft/sum(weights))
    for d in range(len(results), 0, -1):
        sum_of = 0
        for v in range(0, d):
            sum_of += v*weights[v]
        total_mean += sum_of/d
        list_of_means.append((sum_of/d))
    chosen_mean = (total_mean/len(results))*max_res
    print("Unshifted Mean", unshifted_mean/len(results))
    intersect = 0
    for n in range(1, len(results), 1):
        prior = results[n-1]
        current = results[n]
        if chosen_mean >= prior and chosen_mean < current:
            intersect = n - 1
            print("Bounds", prior, current)
    print("Intersection", intersect)
    plt.axvline(x=intersect, color='r')
    #plt.axhline(y=chosen_mean, color='g')
    final_mean = []
    for m in list_of_means:
        #plt.axhline(y=m*max_res, color='y')
        for t in range(1, len(results), 1):
            prior = results[t - 1]
            current = results[t]
            y_val = m*max_res
            if y_val >= prior and y_val < current:
                slope = abs(current - prior)
                position = int(t/slope)
                #plt.axvline(x=(t-1), color='b')
                final_mean.append(t-1)
    print("Mean of means", sum(final_mean)/len(final_mean))
    #plt.axvline(x=chosen_mean, color='r')
    plt.plot(x_axis, results, color='k')
    plt.show()

def low_thresh_test():
    input_paths = ["C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\"]
    files = [input_path + f for input_path in input_paths for f in listdir(input_path) if isfile(join(input_path, f))]
    low_thresh_testing_dict = {}
    for i in files:
        img = io.imread(i)
        filter_results = other_threshold_results(img)
        counts, centers = histogram(img, nbins=256)
        counts = counts[1:]
        centers = centers[1:]
        log_counts2 = np.where(counts != 0, np.log2(counts), 0)
        log_counts10 = np.where(counts != 0, np.log10(counts), 0)
        sample_name = i.split('\\')[-1].split("Noise")[0] + ".tif"
        online_dict = {}
        for online in [False]:
            style_dict = {}
            for style in ["interp1d"]:
                sens_dict = {}
                for S in range(2, 3, 1):
                    s = S/10
                    knee_finder = KneeLocator(x=centers, y=counts, curve="convex", direction="decreasing", S=s, interp_method=style, online=online)
                    knee_finder_log2 = KneeLocator(x=centers, y=log_counts2, curve="convex", direction="decreasing", S=s, interp_method=style, online=online)
                    knee_finder_log10 = KneeLocator(x=centers, y=log_counts10, curve="convex", direction="decreasing", S=s, interp_method=style, online=online)
                    sens_dict[s] = {"Normal": to_str(list(knee_finder.all_knees)), "Log2":to_str(list(knee_finder_log2.all_knees)),
                                    "Log10":to_str(list(knee_finder_log10.all_knees)),
                                    "Manual":[str(manual_Hysteresis[sample_name][1][0]*255), str(manual_Hysteresis[sample_name][1][1]*255)],
                                    "Other Filters":filter_results}
                    '''knee, first_knee, __ = testing_knee(img, style, online, s, filter_results["Otsu"], False)
                    log_knee, first_log, ___ = testing_knee(img, style, online, s, filter_results["Otsu"], True)
                    sens_dict[s] = {"Normal": [str(knee), str(first_knee)], "Log": [str(log_knee), str(first_log)]}'''
                style_dict[style] = sens_dict
            online_dict[int(online)] = style_dict
        #online_dict["Other Filters"] = filter_results
        low_thresh_testing_dict[sample_name] = online_dict
    print(low_thresh_testing_dict)
    with open("C:\\RESEARCH\\Mitophagy_data\\Low_thresh_test_results\\Low_Thresh_Results.json", 'w') as j:
        json.dump(low_thresh_testing_dict, j)
        print("Saved")

def to_str(list_items):
    str_list = []
    for l in list_items:
        str_list.append(str(l))
    return str_list

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


def testing_knee(img, interp_style, online, sensitivity=1, filter_value=None, log_hist=False):
    #print("Histogram for knee")
    counts, centers = histogram(img, nbins=256)
    counts = counts[1:]
    centers = centers[1:]
    gaussian_image = gaussian(img)
    rescaled_gaussian_image = (gaussian_image/np.max(gaussian_image))*np.max(img)
    #print("Rescaled Gaussian Otsu", threshold_otsu(rescaled_gaussian_image))
    norm_gaussian = (gaussian_image/np.max(gaussian_image))*np.max(img)
    gaussian_counts, gaussian_centers = histogram(norm_gaussian, nbins=256)
    gaussian_counts, gaussian_centers = ((gaussian_counts/np.max(gaussian_counts))*np.max(counts)).astype('int'), ((gaussian_centers / np.max(gaussian_centers)) * np.max(centers)).astype('int')

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
        locator = KneeLocator(x=centers, y=counts, S=sensitivity, curve="convex", direction="decreasing", interp_method=interp_style, online=online)
        glocator = KneeLocator(x=gaussian_centers, y=gaussian_counts, S=sensitivity, curve="convex", direction="decreasing", interp_method=interp_style, online=online)
        knee = int(locator.knee)
        gaussian_knee = int(glocator.knee)
        if knee > otsu_thresh and knee_found:
            true_knee = knee
            knee_found = False
            #print("True Knee", true_knee)
        if knee <= otsu_thresh or gaussian_knee < gaussian_otsu:
            '''print("Determined knee", knee, gaussian_knee)
            print("Standard Intensity", centers[0])
            print("Gaussian Intensity", gaussian_centers[0])
            print("Otsu Thresh", otsu_thresh)'''
            centers = centers[1:]
            counts = counts[1:]
            gaussian_centers = gaussian_centers[1:]
            gaussian_counts = gaussian_counts[1:]
        else:
            safe_knee = False
            #print("Final Standard Intensity", centers[0])
            #print("Final Gaussian Intensity", gaussian_centers[0])
            #print("Determined knee", knee, gaussian_knee)
    '''if not knee_found:
        print("Not Found")
        first_knee = locator.knee'''
    gaussian_knee = glocator.knee
    #print("knees: ", locator.all_knees, glocator.all_knees)


    #locator.plot_knee()
    #plt.show()
    #knee = int(locator.norm_knee*255)

    return true_knee, first_knee, otsu_thresh

def testing_knee2(img, interp_style, online, sensitivity=1, filter_value=None, log_hist=False):

    counts, centers = histogram(img, nbins=256)

    if log_hist:
        counts = np.where(counts != 0, np.log10(counts), 0)

    safe_knee = True
    if filter_value is None:
        otsu_thresh = threshold_otsu(img)
    else:
        otsu_thresh = filter_value
    true_knee = 0
    knee_found = True
    first_knee = int(KneeLocator(x=centers, y=counts, S=sensitivity, curve="convex", direction="decreasing", interp_method=interp_style, online=online).knee)
    while safe_knee:
        knee = int(KneeLocator(x=centers, y=counts, S=sensitivity, curve="convex", direction="decreasing", interp_method=interp_style, online=online).knee)
        if knee > otsu_thresh and knee_found:
            true_knee = knee
            knee_found = False

        if knee <= otsu_thresh:
            centers = centers[1:]
            counts = counts[1:]

        else:
            safe_knee = False


    return true_knee, first_knee, otsu_thresh

def tunable_method(slope, log_slope):
    maxes = []
    for s in range(1, len(slope)):
        maxes.append(slope.index(max(slope[:s])))
    maxes = list(set(maxes))
    maxes.sort()
    print("All maxes", maxes)
    av_max = sum(maxes)/len(maxes)
    cluster_seperators = find_cluster(maxes, len(slope))
    if len(cluster_seperators) > 1:
        refine_clusters(cluster_seperators, slope, log_slope)
    fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
    ax1.plot(np.linspace(0, len(slope), len(slope)), slope, color='r')
    ax1.axvline(x=av_max, color='g')
    for t in maxes:
        ax1.axvline(x=t, color='b')
        ax2.axvline(x=t, color='b')
    for g in cluster_seperators:
        ax1.axvline(x=g, color='k')
        ax2.axvline(x=g, color='k')
    ax2.plot(np.linspace(0, len(log_slope), len(log_slope)), log_slope, color='r')
    plt.show()

def find_cluster(max, range_of_values):
    max.reverse()
    print(max)
    between_max_diff = 0
    clusters_distances = []
    clusters = [] # A tuple will be used for the cluster boundaries and the overall size will be assigned to it
    cluster_sep = []
    relative_distance = []
    clusters.append(max[0])
    max_differences = []
    for n in range(1, len(max), 1):
        between_max_diff = max[n-1] - max[n]
        max_differences.append(between_max_diff)
    max_av_distance = sum(max_differences)/len(max_differences)

    for m in range(1, len(max), 1):
        between_max_diff = max[m-1] - max[m]
        clusters_distances.append(between_max_diff)
        relative_distance.append(between_max_diff/range_of_values)
        if len(clusters_distances) > 1:
            rolling = sum(clusters_distances)/len(clusters_distances)
            rolling_prior = sum(clusters_distances[:-1])/(len(clusters_distances)-1)
            #print("Differences", rolling, rolling_prior)
            if rolling > rolling_prior*1.5 or between_max_diff > math.ceil(max_av_distance):
                cluster_sep.append(max[m] + (max[m-1]-max[m])/2)
                clusters.append(max[m])
                #print("Distances", clusters_distances)
                clusters_distances = []
    if len(clusters) == 0:
        clusters = [max[0]]
    clusters.reverse()
    max.reverse()
    return clusters

def refine_clusters(cluster_points, slope, log_slope):
    print("Clusters", cluster_points)
    print("Av Cluster", sum(cluster_points)/len(cluster_points))
    average_slope = []
    iter_slopes = iter(cluster_points)
    cluster_top = next(iter_slopes)
    slope_sum = 0
    log_intercepts = []
    slope_of_slopes = []
    slope_start = 0
    for n in range(len(log_slope)):
        if n == cluster_top:
            begin = log_slope[slope_start]
            end = log_slope[cluster_top]
            slope_of_slopes.append((end-begin)/(cluster_top-slope_start))
            slope_start = cluster_top
        if n <= cluster_top:
            slope_sum += log_slope[n]
        else:
            if n > cluster_top:
                print("Log Values", log_slope[n-1], log_slope[n-2])
                m = (log_slope[n-1] - log_slope[n-2])
                log_inter = m*(cluster_top - n + 2) + log_slope[n-2]
            else:
                log_inter = log_slope[n]
            log_intercepts.append(log_inter)
            average_slope.append(slope_sum/(n-1))
            slope_sum = log_slope[n]
            try:
                cluster_top = next(iter_slopes)
            except StopIteration:
                break
    print("Average Slopes", average_slope)
    percentage_average = []
    change_in_slope = []
    for a in range(1, len(average_slope)):
        percentage_average.append((a, average_slope[a]/average_slope[a-1]))
        change_in_slope.append((a, (slope_of_slopes[a] / slope_of_slopes[a-1])))
        #print("Percentage difference", a-1, "and", a, "is", (average_slope[a]/average_slope[a-1])*100, "Change in slope from", a - 1, "to", a, ":", (slope_of_slopes[a] / slope_of_slopes[a - 1]) * 100)
    percentage_average = sorted(percentage_average, key=lambda x: x[1])
    change_in_slope = sorted(change_in_slope, key=lambda x: x[1])
    rankings = {}
    #print(percentage_average, change_in_slope)
    for t in range(len(percentage_average)):
        rankings[t+1] = 0
    for y in range(len(percentage_average)):
        #print("Average", percentage_average[y][0], "Slope", change_in_slope[y][0])
        rankings[percentage_average[y][0]] += y + 1
        rankings[change_in_slope[y][0]] += y + 1
        #print(rankings)
    #print("Log Intercepts", log_intercepts)
    ordered_rankings = sorted(rankings, key=rankings.get)
    cutoff_max = list(ordered_rankings)[0] + 1
    print("Cutoff Max", cutoff_max - 1)
    print(average_slope[:cutoff_max])
    better_cutoff = average_slope.index(max(average_slope[:cutoff_max]))
    print("Better Cutoff:", better_cutoff)
    max_log_index = average_slope.index(max(average_slope))
    print("Max in range:", log_slope[cluster_points[better_cutoff]])
    print("Average across segment:", sum(log_slope[:cluster_points[better_cutoff]])/len(log_slope[:cluster_points[better_cutoff]]))
    print("Min across segment:", min(log_slope[:cluster_points[better_cutoff]]))
    #print(cluster_points[max_log_index])
    #print("Slopes of slopes", slope_of_slopes)

def to_mip(image_path, low_thresh, high_thresh, manual_params, name):
    save_name = name.split(".")[0]
    save_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisPreview\\MIP\\"
    img = io.imread(image_path)
    #io.imshow(img[0])
    thresholded_image = apply_hysteresis_threshold(img, low_thresh, high_thresh).astype('int')
    masked_image = np.amax(img*thresholded_image, axis=0)
    mip_orig = np.amax(img, axis=0)
    richard_high = manual_params[0][1] * 255
    richard_low = manual_params[0][0] * 255
    richard_thresholded = apply_hysteresis_threshold(img, richard_low, richard_high).astype('int')
    richard_mip = np.amax(img*richard_thresholded, axis=0)
    rensu_high = manual_params[1][1] * 255
    rensu_low = manual_params[1][0] * 255
    rensu_thresholded = apply_hysteresis_threshold(img, rensu_low, rensu_high).astype('int')
    rensu_mip = np.amax(img * rensu_thresholded, axis=0)
    #plt.subplot(2, 2, 1)
    #plt.imshow(mip_orig)
    plt.imsave(save_path + save_name + "Orig.png", mip_orig)
    #plt.title("Original")
    zero_array = np.zeros_like(mip_orig)
    rgb_img = np.stack((mip_orig, masked_image, zero_array), axis=-1)
    io.imshow(rgb_img)
    plt.subplot(2, 2, 2)
    plt.imshow(masked_image)
    plt.imsave(save_path + save_name + "Auto.png", masked_image)
    plt.title("Auto")
    plt.subplot(2, 2, 3)
    plt.imshow(rensu_mip)
    plt.imsave(save_path + save_name + "Rensu.png", rensu_mip)
    plt.title("Rensu")
    plt.subplot(2, 2, 4)
    plt.imshow(richard_mip)
    plt.imsave(save_path + save_name + "Rich.png", richard_mip)
    plt.title("Richard")
    plt.show()
    rgb_img = np.stack((rensu_mip, richard_mip, masked_image), axis=-1).astype('uint8')
    plt.imsave(save_path + save_name + "RGB.png", rgb_img)

def voxel_weight_mip(image_path, low_threshes, high_threshes, name):
    save_name = name.split(".")[0]
    save_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisPreview\\VoxelWeightMIP\\"
    img = io.imread(image_path)
    #io.imshow(img[0])
    masked_image = []
    for l in range(len(low_threshes)):
        thresholded_image = apply_hysteresis_threshold(img, low_threshes[l], high_threshes[l]).astype('int')
        masked_image.append(np.amax(img*thresholded_image, axis=0))
    mip_orig = np.amax(img, axis=0)
    #plt.subplot(2, 2, 1)
    #plt.imshow(mip_orig)
    #plt.imsave(save_path + save_name + "Orig.png", mip_orig)
    #plt.title("Original")
    zero_array = np.zeros_like(mip_orig)
    if len(masked_image) < 3:
        mask_count = len(masked_image)
        for t in range(3 - mask_count):
            masked_image.append(zero_array)
    elif len(masked_image) > 3:
        masked_image = masked_image[:3]
    rgb_img = np.stack(masked_image, axis=-1).astype('uint8')
    plt.imsave(save_path + save_name + "Power.png", rgb_img)

def voxel_weight_stack(image_path, low_threshes, high_threshes, name):
    save_name = name.split(".")[0]
    save_path = "C:\\RESEARCH\\Mitophagy_data\\HysteresisPreview\\VoxelWeightStack\\"
    img = io.imread(image_path)
    #io.imshow(img[0])
    masked_image = []
    for l in range(len(low_threshes)):
        thresholded_image = apply_hysteresis_threshold(img, low_threshes[l], high_threshes[l]).astype('int')
        masked_image.append(img*thresholded_image)

    zero_array = np.zeros_like(img)
    if len(masked_image) < 3:
        mask_count = len(masked_image)
        for t in range(3 - mask_count):
            masked_image.append(zero_array)
    elif len(masked_image) > 3:
        masked_image = masked_image[:3]
    rgb_img = np.stack(masked_image, axis=-1).astype('uint8')
    tifffile.imwrite(save_path + save_name + "Power.tif", rgb_img, imagej=True)


if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Output\\"]
    correct_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Data Results 3\\"
    #generate_histogram()
    #biased_mean_test()
    evaluate_cumulative_hys()
    #image_stitcher(input_path)
    #sample_specific_metrics, sample_variant_metrics = collate_data(input_path, {"C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\":"C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Output\\":"C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\"})
    #save_data2(correct_path, sample_specific_metrics, sample_variant_metrics)
    #loaded_data = load_data(correct_path)
    #intensity_dict = loaded_data["Intensity"]
    #blocked_prec = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19]
    #intensity_graphs3(intensity_dict, "Mean", 1)
    #thresholded_image_view("C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\", "CCCP_1C=0.tif", 53, 173)
    #filters = loaded_data["Filter"]
    #errors = loaded_data["Error"]
    #filter_graphing(filters, errors)
    #low_thresh_test()
    '''Result is that Li, Triangle and Mean are equally low in error. Any can be used for the filter choice atm but this is only for graphing the range of 
    intensities as noise response is unknown and there is no control'''
    '''df = pd.DataFrame(sample_variant_metrics)
    print(df.columns)
    other_df = df[["Intensity Range", "Sample", "Filter Type", "Precision", "Starting Density"]]
    print(other_df)
    check_df = df[["Sample", "Filter Type", "Filter_Thresh", "Precision", "Starting Density", "Low_Thresh", "High_Thresh"]]'''



