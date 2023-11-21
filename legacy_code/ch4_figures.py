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
import tifffile
from scipy import ndimage as ndi
import seaborn as sns
from mayavi import mlab
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def slice_images(image_list, destination):
    for im in image_list:
        image = io.imread(im)
        depth = image.shape[0]
        middle_slice = int(depth/2)
        io.imsave(destination[im], image[middle_slice])

def get_noise_metrics(noise_path):
    '''
    This function is to extract the json stored noise metrics and consolidate them
    :param noise_path:
    :return:
    '''
    with open(noise_path, "r") as j:
        noise_metrics = json.load(j)
    df = pd.DataFrame(noise_metrics)
    df_trimmed = df.drop(df[(df['Poisson'] == 0) & (df['Gaussian'] == 0.000)].index)
    poisson_values = df_trimmed['Poisson'].unique()
    gaussian_values = df_trimmed['Gaussian'].unique()
    consolidated_measures = {"Gaussian":[], "Poisson":[], "SNR_Mean":[], "SNR_Var":[], "SSIM_Mean":[], "SSIM_Var":[]}
    for p in poisson_values:
        for g in gaussian_values:
            if p != 0 or g != 0:
                selected_df = df[(df['Poisson'] == p) & (df['Gaussian'] == g)]
                mean_snr = selected_df['SNR'].mean()
                var_snr = selected_df['SNR'].std()
                mean_ssim = selected_df['SSIM'].mean()
                var_ssim = selected_df['SSIM'].std()


                consolidated_measures["Gaussian"].append(g)
                consolidated_measures["Poisson"].append(p)
                consolidated_measures["SNR_Mean"].append(mean_snr)
                consolidated_measures["SNR_Var"].append(var_snr)
                consolidated_measures["SSIM_Mean"].append(mean_ssim)
                consolidated_measures["SSIM_Var"].append(var_ssim)
    consolidated_measures = pd.DataFrame(consolidated_measures)

    return consolidated_measures


def prepare_2d_array_from_df(axis_names, df):
    row_values = df.to_dict('records')
    x_axis_values, y_axis_values = list(set(df[axis_names[0]])), list(set(df[axis_names[1]]))
    x_axis_values.sort()
    y_axis_values.sort()
    x_axis = {x_axis_values[k]: k for k in range(len(x_axis_values))}
    y_axis = {y_axis_values[k]: k for k in range(len(y_axis_values))}
    data_keys = list(set(df.columns).difference(axis_names))
    data_tables = {}
    for dk in data_keys:
        data_tables[dk] = np.zeros(tuple([len(x_axis), len(y_axis)]))
    for rv in row_values:
        for dk in data_keys:
            x_coord = x_axis[rv[axis_names[0]]]
            y_coord = y_axis[rv[axis_names[1]]]
            data_tables[dk][x_coord, y_coord] = rv[dk]

    return data_tables


def generate_surfaces(x_values, y_values, params, title=None):
    '''
    This function will accept x-axis and y-axis values for the integers and then the parameters to build it.
    Expected to be single entry dictionaries for the key to be used for the axis name and the value to be a list
    of values
    :param x_values:
    :param y_values:
    :param params:
    :return:
    '''
    def var_name(name):
        if name.endswith("_Var"):
            name = name[:-4] + " Std"
            return name
        else:
            return name
    x_axis_label = list(x_values)[0]
    x_axis_values = list(set(x_values[x_axis_label]))
    y_axis_label = list(y_values)[0]
    y_axis_values = list(set(y_values[y_axis_label]))
    x_axis_values.sort()
    y_axis_values.sort()
    X, Y = np.meshgrid(np.array(y_axis_values), np.array(x_axis_values),)
    for k, v in params.items():
        if k == "SNR" or k == "SNR_Var":
            print(k)
            #change inf to nan
            Z = v
            Z[Z == np.inf] = np.nan
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)
            #Z[0, 0] = np.nan
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            plt.rcParams.update({'font.size': 14})
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_xlabel("Gaussian")
            ax.set_ylabel("Poisson")
            #ax.set_zlabel(var_name(k))
            '''if title is not None:
                fig.suptitle(title)'''
            plt.show()

def consolidate_thresh_metrics(thresh_data):
    '''
    This function will be handed a trimmed DataFrame where the Method has already been used as a selector
    :param thresh_data: DataFrame
    :return:
    '''
    df = pd.DataFrame(thresh_data)
    methods = df["Method"].unique()
    #df_trimmed = df.drop(df[(df['P'] == 0) & (df['G'] == 0.000)].index)
    df_trimmed = df
    poisson_values = df_trimmed['P'].unique()
    gaussian_values = df_trimmed['G'].unique()
    method_separated = {}
    for m in methods:
        consolidated_measures = {"Gaussian": [], "Poisson": [], "Accuracy": [], "Accuracy_Var": [],
                                 "Precision": [], "Precision_Var": [], "Recall": [], "Recall_Var": [], "SNR": [],
                                 "SNR_Var": [], "SSIM": [], "SSIM_Var": []}
        method_df = df[(df['Method'] == m)]
        for p in poisson_values:
            for g in gaussian_values:
                if p != 0 or g != 0:
                    selected_df = method_df[(method_df['P'] == p) & (method_df['G'] == g)]
                    mean_snr = selected_df['SNR'].mean()
                    var_snr = selected_df['SNR'].std()
                    mean_ssim = selected_df['SSIM'].mean()
                    var_ssim = selected_df['SSIM'].std()
                    mean_acc = selected_df['Accuracy'].mean()
                    var_acc = selected_df['Accuracy'].std()
                    mean_prec = selected_df['Precision'].mean()
                    var_prec = selected_df['Precision'].std()
                    mean_rec = selected_df['Recall'].mean()
                    var_rec = selected_df['Recall'].std()

                    consolidated_measures["Gaussian"].append(float(g))
                    consolidated_measures["Poisson"].append(float(p))
                    consolidated_measures["Accuracy"].append(mean_acc)
                    consolidated_measures["Accuracy_Var"].append(var_acc)
                    consolidated_measures["Precision"].append(mean_prec)
                    consolidated_measures["Precision_Var"].append(var_prec)
                    consolidated_measures["Recall"].append(mean_rec)
                    consolidated_measures["Recall_Var"].append(var_rec)
                    consolidated_measures["SNR"].append(mean_snr)
                    consolidated_measures["SNR_Var"].append(var_snr)
                    consolidated_measures["SSIM"].append(mean_ssim)
                    consolidated_measures["SSIM_Var"].append(var_ssim)
        method_separated[m] = pd.DataFrame(consolidated_measures)

    return method_separated

def consolidate_ihh_metrics(thresh_data):
    '''
    This function will be handed a trimmed DataFrame where the Method has already been used as a selector
    :param thresh_data: DataFrame
    :return:
    '''
    df = pd.DataFrame(thresh_data)
    voxel = df["Voxel"].unique()
    window = df["Window"].unique()
    #df_trimmed = df.drop(df[(df['P'] == 0) & (df['G'] == 0.000)].index)
    df_trimmed = df
    poisson_values = df_trimmed['P'].unique()
    gaussian_values = df_trimmed['G'].unique()
    voxel_separated = {}
    for v in voxel:
        window_separated = {}
        for w in window:
            consolidated_measures = {"Gaussian": [], "Poisson": [], "Accuracy": [], "Accuracy_Var": [],
                                     "Precision": [], "Precision_Var": [], "Recall": [], "Recall_Var": [], "SNR": [],
                                     "SNR_Var": [], "SSIM": [], "SSIM_Var": []}
            chosen_df = df[(df['Voxel'] == v) & (df['Window'] == w)]
            for p in poisson_values:
                for g in gaussian_values:
                    selected_df = chosen_df[(chosen_df['P'] == p) & (chosen_df['G'] == g)]
                    mean_snr = selected_df['SNR'].mean()
                    var_snr = selected_df['SNR'].std()
                    mean_ssim = selected_df['SSIM'].mean()
                    var_ssim = selected_df['SSIM'].std()
                    mean_acc = selected_df['Accuracy'].mean()
                    var_acc = selected_df['Accuracy'].std()
                    mean_prec = selected_df['Precision'].mean()
                    var_prec = selected_df['Precision'].std()
                    mean_rec = selected_df['Recall'].mean()
                    var_rec = selected_df['Recall'].std()

                    consolidated_measures["Gaussian"].append(float(g))
                    consolidated_measures["Poisson"].append(float(p))
                    consolidated_measures["Accuracy"].append(mean_acc)
                    consolidated_measures["Accuracy_Var"].append(var_acc)
                    consolidated_measures["Precision"].append(mean_prec)
                    consolidated_measures["Precision_Var"].append(var_prec)
                    consolidated_measures["Recall"].append(mean_rec)
                    consolidated_measures["Recall_Var"].append(var_rec)
                    consolidated_measures["SNR"].append(mean_snr)
                    consolidated_measures["SNR_Var"].append(var_snr)
                    consolidated_measures["SSIM"].append(mean_ssim)
                    consolidated_measures["SSIM_Var"].append(var_ssim)
            window_separated[w] = pd.DataFrame(consolidated_measures)
        voxel_separated[v] = window_separated

    return voxel_separated


def plot_threshold_surfaces(thresh_path):
    with open(thresh_path, "r") as j:
        thresh_metrics = json.load(j)

    method_dfs = consolidate_thresh_metrics(thresh_metrics)

    for m, thresh_df in method_dfs.items():
        print("Method", m)
        #trimmed_df = thresh_df[["Gaussian", "Poisson", "Accuracy"]]
        flattened_data = prepare_2d_array_from_df(["Gaussian", "Poisson"], thresh_df)
        #For SNR and SSIM there are nan values in the array. These need to be trimmed for this beforehand
        x_axis = {"Gaussian": np.sort(thresh_df["Gaussian"].unique())}
        y_axis = {"Poisson": np.sort(thresh_df["Poisson"].unique())}
        generate_surfaces(x_axis, y_axis, flattened_data, m)


def plot_noise_metric_surfaces(noise_df):
    flattened_data = prepare_2d_array_from_df(["Gaussian", "Poisson"], noise_df)

    print(flattened_data['SSIM_Mean'])
    x_axis = {"Gaussian": np.sort(noise_df["Gaussian"].unique())}
    y_axis = {"Poisson": np.sort(noise_df["Poisson"].unique())}
    print(len(x_axis["Gaussian"]), len(y_axis["Poisson"]), flattened_data['SSIM_Mean'].shape)
    generate_surfaces(x_axis, y_axis, flattened_data)

def plot_ihh_surfaces(ihh_path):
    with open(ihh_path, "r") as j:
        ihh_metrics = json.load(j)
    voxel_dfs = consolidate_ihh_metrics(ihh_metrics)
    for v, windows in voxel_dfs.items():
        print("Voxel bias", v)
        for w, ihh_df in windows.items():
            print("Window", w)
            title = "IHH bias: " + str(v) + " Window bias: " + str(w)
            flattened_data = prepare_2d_array_from_df(["Gaussian", "Poisson"], ihh_df)
            x_axis = {"Gaussian": np.sort(ihh_df["Gaussian"].unique())}
            y_axis = {"Poisson": np.sort(ihh_df["Poisson"].unique())}
            generate_surfaces(x_axis, y_axis, flattened_data, title)

def export_data_for_table(data_path, data_name):

    with open(join(data_path, data_name), "r") as j:
        data_metrics = json.load(j)

    metric_list = ["Accuracy", "Precision", "Recall", "SNR", "SSIM"]

    def restructure_thresh_data(json_data, metric_name):
        df = pd.DataFrame(json_data)
        poisson_values = df['P'].unique()
        gaussian_values = df['G'].unique()
        methods = df['Method'].unique()
        consolidated_measures = {"Method": [], "Gaussian": [], "Poisson": [], "Mean": [], "Std": []}
        for p in poisson_values:
            for g in gaussian_values:
                for m in methods:
                    selected_df = df[(df['P'] == p) & (df['G'] == g) & (df['Method'] == m)]
                    consolidated_measures["Gaussian"].append(g)
                    consolidated_measures["Poisson"].append(p)
                    consolidated_measures["Method"].append(m)
                    consolidated_measures["Mean"].append(selected_df[metric_name].mean())
                    consolidated_measures["Std"].append(selected_df[metric_name].std())
        return pd.DataFrame(consolidated_measures)

    def restructure_ihh_data(json_data, metric_name):
        df = pd.DataFrame(json_data)
        poisson_values = df['P'].unique()
        gaussian_values = df['G'].unique()
        ihh_bias = df["Voxel"].unique()
        window_bias = df["Window"].unique()
        consolidated_measures = {"Gaussian": [], "Poisson": [], "IHH": [], "Window": [], "Mean": [],
                                 "Std": []}
        for p in poisson_values:
            for g in gaussian_values:
                for v in ihh_bias:
                    for w in window_bias:
                        selected_df = df[(df['P'] == p) & (df['G'] == g) & (df['Voxel'] == v) & (df['Window'] == w)]
                        consolidated_measures["Gaussian"].append(g)
                        consolidated_measures["Poisson"].append(p)
                        consolidated_measures["IHH"].append(v)
                        consolidated_measures["Window"].append(w)
                        consolidated_measures["Mean"].append(selected_df[metric_name].mean())
                        consolidated_measures["Std"].append(selected_df[metric_name].std())
        return pd.DataFrame(consolidated_measures)

    thresh_or_not = "Method" in list(data_metrics)

    for m in metric_list:
        if thresh_or_not:
            # for global or local thresh
            table_df = restructure_thresh_data(data_metrics, m)
        else:
            # for IHH data
            table_df = restructure_ihh_data(data_metrics, m)
        csv_name = data_name.split('.')[0] + '_' + m + '.csv'
        table_df.to_csv(join(data_path, csv_name))


if __name__ == "__main__":
    image_names = ["F:\\clean images\\Blurred\\CCCP_1C=1T=0.tif",
                   "F:\\clean images\\Binarized\\Hysteresis\\CCCP_1C=1T=0.tif"]
    save_point = {"F:\\clean images\\Blurred\\CCCP_1C=1T=0.tif":"F:\\clean images\\Figures\\middle_slice_blurred.png",
             "F:\\clean images\\Binarized\\Hysteresis\\CCCP_1C=1T=0.tif":"F:\\clean images\\Figures\\middle_slice.png"}
    n_path = "F:\\clean images\\Backup\\Noise_metrics.json"
    th_path = "F:\\clean images\\Noise Applied\\"
    #export_data_for_table(th_path, "ihh_results.json")
    plot_ihh_surfaces(join(th_path, "ihh_results.json"))
    #plot_threshold_surfaces(join(th_path, "global_thresh.json"))
    print("Local thresholds")
    #plot_threshold_surfaces(join(th_path, "local_thresh.json"))
    #plot_noise_metric_surfaces(get_noise_metrics(n_path))