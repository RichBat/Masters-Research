import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold
from scipy import interpolate
import tifffile

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
    holder = []
    for c in list_of_strings:
        holder.append(float(c))
    return holder

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
            #print(i["Sample"])
            for c in ["Fixed", "AdjDensity", "FixedDensity"]:
                if i["Intensity Range"][c] is not None:
                    intens_range_dict[c] = i["Intensity Range"][c],
                    for l in ["Unlooped"]: #Removed Looped as it was removed to save processing time
                        temp_dict[(l, c)] = {"Scaled Voxels per intensity": convert_to_float(i["Scaled Voxels per intensity"][l + " - " + c]),
                                            "Voxels per intensity": convert_to_float(i["Voxels per intensity"][l + " - " + c]),
                                            "Voxel Changes": convert_to_float(i["Voxel Changes"][l + " - " + c]), "High_Thesh": i["High_Thesh"][l + " - " + c]}
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
        for j in v:
            prec = j["Precision"]
            if prec not in blocked_prec:
                image_max = j["Image Max"]
                low_thresh[prec] = j["Low_Thresh"]
                total_v = j["Total Voxels"]
                intens_range = j["Intensity Range"]  # Dictionary of three intensity range variations
                # print("Stuff", intens_range)
                intensities_per_prec[prec] = intens_range["FixedDensity"][0]
                max_intens[prec] = max(intens_range["AdjDensity"][0])
                varied_outputs = {"Unlooped": {}, "Looped": {}}
                intensity_range_configs = ["AdjDensity", "FixedDensity"]  # Removed "Fixed" as it is not in the latest data
                for t in intensity_range_configs:
                    varied_outputs["Unlooped"][t] = j["Varied Outputs"][("Unlooped", t)]
                    # varied_outputs["Looped"][t] = j["Varied Outputs"][("Looped", t)]
                # Going to use just unlooped first
                intens_to_be_used = intens_range["FixedDensity"][0]
                print("Intensity Range:", intens_to_be_used)
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
                    print(prior_step, current_step)
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
                print(voxel_per_prec_val)
                print(rescaled_voxel_diffs)
                for rvd in range(1, len(rescaled_voxel_diffs)):
                    top_value = rescaled_voxel_diffs[rvd] if rescaled_voxel_diffs[rvd] != 0 else 0.000000000000000000000000000000000000000000000001
                    bottom_value = rescaled_voxel_diffs[rvd - 1] if rescaled_voxel_diffs[rvd - 1] != 0 else 0.000000000000000000000000000000000000000000000001
                    rescaled_changes.append(top_value / bottom_value)
                # print("Rescaled Voxle Diffs:", rescaled_voxel_diffs)
                # print("Rescaled Changes:", rescaled_changes)
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
                # intensity_to_be_used[prec] = intens_to_be_used[scaled_vox_average_index]
                ax1.plot(intens_to_be_used, rescaled_voxel_diffs, '-D', label=str(j["Precision"]))
                ax2.plot(intens_to_be_used, voxel_per_prec_val, '-D', label=str(j["Precision"]))
                ax3.plot(intens_to_be_used[:-1], calculated_changes, '-D', label=str(j["Precision"]))
                colour = plt.gca().lines[-1].get_color()
                # ax1.axhline(y=scaled_vox_average, xmin=0, xmax=1, color=colour)
                # ax1.axvline(x=scaled_vox_average_intensity, ymin=0, ymax=1, color=colour)
                # ax2.axhline(y=prec_rescaled_average, xmin=0, xmax=1, color=colour)
                # ax2.axvline(x=intens_to_be_used[scaled_vox_prec_average_index], ymin=0, ymax=1, color=colour)
                # ax3.axhline(y=calc_change_average, xmin=0, xmax=1, color=colour)
                # ax3.axvline(x=inters_ave, ymin=0, ymax=1, color=colour)
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
        across_samples_thresholds[sample_name] = {"Richard":{"Low:":manual_Hysteresis[sample_name][0][0] * 255, "High:":manual_Hysteresis[sample_name][0][1] * 255},
                                                  "Rensu":{"Low:":manual_Hysteresis[sample_name][1][0] * 255, "High:":manual_Hysteresis[sample_name][1][1] * 255},
                                                  "Auto":{"Low:":low_thresh[0.02], "Ave Rescaled:":mean_rescaled_intense, "Old Ave Rescaled:":old_mean_rescaled_intense,
                                                          "Change Intensity:":intersect_high}}
        '''errors1 = calculate_error_values(sample_name, low_thresh[0.02], mean_rescaled_intense, image, False)
        errors2 = calculate_error_values(sample_name, low_thresh[0.02], intersect_high, image, False)
        print("Errors with Rescaled:", errors1)
        print("Errors with Intersect:", errors2)
        per_sample_error[sample_name] = {"Rescale Errors:": errors1, "Intersect Errors": errors2}'''
        ax1.set_title("Rescaled Voxels per intensity")
        ax2.set_title("Voxels per intensity")
        ax3.set_title("Rescaled Voxel Changes")
        ax1.axvline(x=mean_rescaled_intense, ymin=0, ymax=1, color='k')
        ax3.axvline(x=intersect_high, ymin=0, ymax=1, color='k')
        plt.legend()
        #plt.show()
    for key, value in across_samples_thresholds.items():
        print("Sample", key)
        print(value)
    #print("Errors across all samples", per_sample_error)


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
    return [richard_error, rensu_error, average_error]

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

if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Output\\"]
    correct_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Data Results 4\\"
    generate_histogram()
    #image_stitcher(input_path)
    #sample_specific_metrics, sample_variant_metrics = collate_data(input_path, {"C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\":"C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\", "C:\\RESEARCH\\Mitophagy_data\\Testing Output\\":"C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\"})
    #save_data2(correct_path, sample_specific_metrics, sample_variant_metrics)
    #loaded_data = load_data(correct_path)
    #intensity_dict = loaded_data["Intensity"]
    #blocked_prec = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19]
    #intensity_graphs3(intensity_dict, "Mean", 1, blocked_prec)
    #thresholded_image_view("C:\\RESEARCH\\Mitophagy_data\\Testing Input data 2\\", "CCCP_1C=0.tif", 53, 173)
    #filters = loaded_data["Filter"]
    #errors = loaded_data["Error"]
    #filter_graphing(filters, errors)
    '''Result is that Li, Triangle and Mean are equally low in error. Any can be used for the filter choice atm but this is only for graphing the range of 
    intensities as noise response is unknown and there is no control'''
    '''df = pd.DataFrame(sample_variant_metrics)
    print(df.columns)
    other_df = df[["Intensity Range", "Sample", "Filter Type", "Precision", "Starting Density"]]
    print(other_df)
    check_df = df[["Sample", "Filter Type", "Filter_Thresh", "Precision", "Starting Density", "Low_Thresh", "High_Thresh"]]'''



