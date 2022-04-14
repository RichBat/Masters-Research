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
            if image_paths is not None:
                if i in list(image_paths):
                    image_path = image_paths[i] + sample.split(".")[0] + "Noise000.tif"
                    img = io.imread(image_path)
            sample_specific_metrics[sample] = {}
            threshold_values = s_metrics["All Filter Thresh"]
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

if __name__ == "__main__":
    input_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\"
    correct_path = "C:\\RESEARCH\\Mitophagy_data\\Testing Data Results\\"
    #sample_specific_metrics, sample_variant_metrics = collate_data(input_path, {"C:\\RESEARCH\\Mitophagy_data\\Testing Output 2\\":"C:\\RESEARCH\\Mitophagy_data\\Testing Input data\\"})
    #save_data(correct_path, sample_specific_metrics, sample_variant_metrics)
    loaded_data = load_data(correct_path)
    intensity_dict = loaded_data["Intensity"]
    intensity_graphs2(intensity_dict, "Mean", 1)
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



