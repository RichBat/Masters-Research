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


def generate_surfaces(x_values, y_values, params):
    '''
    This function will accept x-axis and y-axis values for the integers and then the parameters to build it.
    Expected to be single entry dictionaries for the key to be used for the axis name and the value to be a list
    of values
    :param x_values:
    :param y_values:
    :param params:
    :return:
    '''
    x_axis_label = list(x_values)[0]
    x_axis_values = list(set(x_values[x_axis_label]))
    y_axis_label = list(y_values)[0]
    y_axis_values = list(set(y_values[y_axis_label]))
    x_axis_values.sort()
    y_axis_values.sort()
    X, Y = np.meshgrid(np.array(y_axis_values), np.array(x_axis_values),)
    for k, v in params.items():
        print(k)
        Z = v
        vmin = Z.min()
        vmax = Z.max()
        Z[0, 0] = np.nan
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("Gaussian")
        ax.set_ylabel("Poisson")
        ax.set_zlabel("SNR")
        plt.show()

def consolidate_thresh_metrics(thresh_data):
    '''
    This function will be handed a trimmed DataFrame where the Method has already been used as a selector
    :param thresh_data: DataFrame
    :return:
    '''
    df = pd.DataFrame(thresh_data)
    df_trimmed = df.drop(df[(df['Poisson'] == 0) & (df['Gaussian'] == 0.000)].index)
    poisson_values = df_trimmed['Poisson'].unique()
    gaussian_values = df_trimmed['Gaussian'].unique()
    consolidated_measures = {"Gaussian": [], "Poisson": [], "Accuracy": [], "Accuracy_Var": [],
                             "Precision": [], "Precision_Var":[], "Recall": [], "Recall_Var": [], "SNR": [],
                             "SNR_Var": [], "SSIM": [], "SSIM_Var": []}
    for p in poisson_values:
        for g in gaussian_values:
            if p != 0 or g != 0:
                selected_df = df[(df['Poisson'] == p) & (df['Gaussian'] == g)]
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

                consolidated_measures["Gaussian"].append(g)
                consolidated_measures["Poisson"].append(p)
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
    consolidated_measures = pd.DataFrame(consolidated_measures)

    return consolidated_measures

def plot_threshold_surfaces(thresh_path):
    with open(thresh_path, "r") as j:
        thresh_metrics = json.load(j)

    thresh_df = pd.DataFrame(thresh_metrics)
    print(thresh_df.columns)

def plot_noise_metric_surfaces(noise_df):
    flattened_data = prepare_2d_array_from_df(["Gaussian", "Poisson"], noise_df)

    print(flattened_data['SSIM_Mean'])
    x_axis = {"Gaussian": np.sort(noise_df["Gaussian"].unique())}
    y_axis = {"Poisson": np.sort(noise_df["Poisson"].unique())}

    generate_surfaces(x_axis, y_axis, flattened_data)


if __name__ == "__main__":
    image_names = ["F:\\clean images\\Blurred\\CCCP_1C=1T=0.tif",
                   "F:\\clean images\\Binarized\\Hysteresis\\CCCP_1C=1T=0.tif"]
    save_point = {"F:\\clean images\\Blurred\\CCCP_1C=1T=0.tif":"F:\\clean images\\Figures\\middle_slice_blurred.png",
             "F:\\clean images\\Binarized\\Hysteresis\\CCCP_1C=1T=0.tif":"F:\\clean images\\Figures\\middle_slice.png"}
    n_path = "F:\\clean images\\Backup\\Noise_metrics.json"
    th_path = "F:\\clean images\\Noise Applied\\"
    #plot_threshold_surfaces(join(th_path, "global_thresh.json"))
    plot_noise_metric_surfaces(get_noise_metrics(n_path))