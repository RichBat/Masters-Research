import numpy as np
import math

import pandas as pd
from skimage.exposure import histogram
from skimage import data, io, util
from scipy import special, stats
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import morphology
import warnings
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json

save_loc = "F:\\clean images\\noise_testing\\"

def apply_poisson(im, scaling, normal=True):
    if normal:
        poisson_applied = np.random.poisson(im * scaling)
        #print("Poisson range", poisson_applied.max())
        noisy_image = np.clip(im + poisson_applied, 0, 255).astype('uint8')
    else:
        noisy_image = (util.random_noise(im*scaling, mode='poisson')*im).astype('uint8')
    return noisy_image


def apply_gaussian(im, scaling, normal=True):
    if normal:
        gauss_layer = np.random.normal(0, scaling, im.shape)
        #print("Gaussian range", gauss_layer.max())
        noisy_image = np.clip(im + gauss_layer, 0, 255).astype('uint8')
    else:
        noisy_image = (util.random_noise(im, mode='gaussian', var=scaling)*im.max()).astype('uint8')

    return noisy_image


def test_poisson_tuning(input_image, scaling=1):
    if type(input_image) is str:
        im = io.imread(input_image).astype('uint8')
    else:
        im = input_image.astype('uint8')
    poisson_applied = np.random.poisson(im*scaling)
    clipped_noise = np.clip(im + poisson_applied, 0, 255).astype('uint8')
    sk_poisson = (util.random_noise(im*scaling, mode='poisson')*im).astype('uint8')
    return clipped_noise, sk_poisson

def test_gaussian_tuning(input_image, scaling=0.01):
    im = io.imread(input_image).astype('uint8')
    centre_slice = int(im.shape[0] / 2)
    gaussian_noise = apply_gaussian(im, scaling)
    io.imshow(gaussian_noise[centre_slice])
    plt.show()
    #io.imsave(join(save_location, "Gaussian_test" + str(scaling) + ".tif"), gaussian_noise)


def compare_noise_approach(im_path, poiss_scalings):
    '''
    This function will take a centre slice of a single image, get the noise applications (not combined)
    and compare whether using the Skimage approach is better.
    :param im_path:
    :param poiss_scalings:
    :param gauss_scalingss:
    :return:
    '''
    im = io.imread(im_path).astype('uint8')
    centre_slice = int(im.shape[0] / 2)
    poiss_concertina = []
    sk_poiss_concertina = []
    for p in poiss_scalings:
        normal_poisson, sk_poisson = test_poisson_tuning(im, p)
        poiss_concertina.append(normal_poisson[centre_slice])
        sk_poiss_concertina.append(sk_poisson[centre_slice])
        io.imsave(join(save_loc, "poisson_test_scale" + str(p) + ".tif"), normal_poisson)
        io.imsave(join(save_loc, "sk_poisson_test_scale" + str(p) + ".tif"), sk_poisson)
    io.imsave(join(save_loc, "normal_poiss_concertina.tif"), np.stack(poiss_concertina, axis=0))
    io.imsave(join(save_loc, "skimage_poiss_concertina.tif"), np.stack(sk_poiss_concertina, axis=0))

def compare_noise_hist(im_path, poiss_scalings):
    '''
    This function will take a centre slice of a single image, get the noise applications (not combined)
    and compare whether using the Skimage approach is better.
    :param im_path:
    :param poiss_scalings:
    :param gauss_scalingss:
    :return:
    '''
    im = io.imread(im_path).astype('uint8')
    normal_hist_set = {}
    skimage_hist_set = {}
    orig_hist, orig_intens = histogram(im)
    normal_hist_set[0] = (orig_hist, orig_intens)
    skimage_hist_set[0] = (orig_hist, orig_intens)
    for p in poiss_scalings:
        normal_poisson, sk_poisson = test_poisson_tuning(im, p)
        normal_hist_set[p] = histogram(normal_poisson)
        skimage_hist_set[p] = histogram(sk_poisson)

    organized_hist = {"intens":[], "count":[], "scale":[]}
    for scale, hist in normal_hist_set.items():
        #norm_factor = (hist[0][1:].max())
        norm_factor = 1
        for t in range(1, hist[0].shape[0]):
            organized_hist["intens"].append(hist[1][t])
            organized_hist["count"].append(hist[0][t]/norm_factor)
            organized_hist["scale"].append(str(scale))
    hist_df = pd.DataFrame(organized_hist)
    sns.lineplot(data=hist_df, x="intens", y="count", hue="scale")
    plt.show()


def noise_ranges(im_path, poiss_scaling, gauss_scaling, save_path=None):
    save_path = "F:\\clean images\\noise_testing\\test_scaling\\" if save_path is None else save_path
    image = io.imread(im_path)
    window_size = image.shape[0] if image.shape[0] % 2 != 0 else image.shape[0] - 1
    # poisson noise
    poisson_readings = {"Scaling": [], "Metric":[], "Value":[]}
    for p in poiss_scaling:
        noisy_outcome = apply_poisson(image, p).astype('uint8')
        poisson_readings["Value"].append(peak_signal_noise_ratio(image, noisy_outcome))
        poisson_readings["Metric"].append("PSNR")
        poisson_readings["Value"].append(structural_similarity(image, noisy_outcome, win_size=window_size))
        poisson_readings["Metric"].append("SSIM")
        poisson_readings["Scaling"].append(p)
        poisson_readings["Scaling"].append(p)
        io.imsave(join(save_path, "Poisson_" + str(p) + ".tif"), noisy_outcome)

    # gaussian noise
    gaussian_readings = {"Scaling": [], "Metric":[], "Value":[]}
    for g in gauss_scaling:
        noisy_outcome = apply_gaussian(image, g).astype('uint8')
        gaussian_readings["Value"].append(peak_signal_noise_ratio(image, noisy_outcome))
        gaussian_readings["Metric"].append("PSNR")
        gaussian_readings["Value"].append(structural_similarity(image, noisy_outcome, win_size=window_size))
        gaussian_readings["Metric"].append("SSIM")
        gaussian_readings["Scaling"].append(g)
        gaussian_readings["Scaling"].append(g)
        io.imsave(join(save_path, "Gaussian_" + str(g) + ".tif"), noisy_outcome)

    with open(join(save_path, "poisson_ranges.json"), 'w') as j:
        json.dump(poisson_readings, j)
    with open(join(save_path, "gaussian_ranges.json"), 'w') as j:
        json.dump(gaussian_readings, j)

def read_noise_results(open_path):

    with open(join(open_path, "poisson_ranges.json"), 'r') as j:
        poisson_readings = json.load(j)
    with open(join(open_path, "gaussian_ranges.json"), 'r') as j:
        gaussian_readings = json.load(j)

    poisson_df = pd.DataFrame(poisson_readings)
    gaussian_df = pd.DataFrame(gaussian_readings)

    #poisson_df["Scaling"] = poisson_df["Scaling"].astype(str)
    gaussian_df["Scaling"] = poisson_df["Scaling"].astype(str)

    '''psnr_gauss_df = poisson_df.loc[poisson_df["Metric"] == "PSNR", ["Scaling", "Value"]]
    ssim_gauss_df = gaussian_df.loc[gaussian_df["Metric"] == "SSIM", ["Scaling", "Value"]]
    psnr_poiss_df = poisson_df.loc[poisson_df["Metric"] == "PSNR", ["Scaling", "Value"]]
    ssim_poiss_df = poisson_df.loc[poisson_df["Metric"] == "SSIM", ["Scaling", "Value"]]'''

    poisson_df.loc[poisson_df["Metric"] == "PSNR", "Value"] = poisson_df.loc[poisson_df["Metric"] == "PSNR", "Value"]/poisson_df.loc[poisson_df["Metric"] == "PSNR", "Value"].max()

    print(poisson_df[poisson_df["Metric"]=="SSIM"])
    plt.figure()
    plt.title("Poisson PSNR")
    sns.barplot(data=poisson_df, x="Scaling", y="Value", hue="Metric")

    gaussian_df.loc[gaussian_df["Metric"] == "PSNR", "Value"] = gaussian_df.loc[gaussian_df["Metric"] == "PSNR", "Value"] / \
                                                              gaussian_df.loc[
                                                                  gaussian_df["Metric"] == "PSNR", "Value"].max()

    plt.figure()
    plt.title("Gaussian Readings")
    sns.barplot(data=gaussian_df, x="Scaling", y="Value", hue="Metric")

    plt.show()

def ssim_testing(test_image, noise_image):
    val, sim_im = structural_similarity(test_image, noise_image, full=True, win_size=5)
    print("Similarity", val)
    for t in range(sim_im.shape[0]):
        plt.figure()
        plt.title("Slice " + str(t))
        io.imshow(sim_im[t])
    plt.show()

def combined_noise_ranges(blurred_im, poisson, gauss, save_location):
    '''
    The blurred image will be proved by the first argument. poisson_params and gauss_params will be 3 element tuples.
    First element will be the starting value, the second element will be the step size, third element will be the
    number of points.
    :param blurred_im:
    :param poisson:
    :param gauss:
    :return:
    '''

    ssim_results = {"Poisson":[], "Gaussian":[], "Value":[]}
    psnr_results = {"Poisson": [], "Gaussian": [], "Value": []}

    for p in np.arange(start=poisson[0], stop=poisson[1], step=poisson[2]):
        for g in np.arange(start=gauss[0], stop=gauss[1], step=gauss[2]):

            poisson_applied = apply_poisson(blurred_im, round(p, 2))
            gaussian_applied = apply_gaussian(poisson_applied, round(g, 2))

            ssim_results["Poisson"].append(round(float(p), 2))
            ssim_results["Gaussian"].append(round(float(g), 2))
            ssim_results["Value"].append(structural_similarity(blurred_im, gaussian_applied, win_size=3).astype(float))

            psnr_results["Poisson"].append(round(float(p), 2))
            psnr_results["Gaussian"].append(round(float(g), 2))
            psnr_results["Value"].append(peak_signal_noise_ratio(blurred_im, gaussian_applied).astype(float))

            new_name = save_location.split("\\")[-2] + "_p" + str(round(p, 2)) + "g" + str(round(g, 2)) + ".tif"
            io.imsave(join(save_location, new_name), gaussian_applied)

            with open(join(save_location, "ssim_results.json"), 'w') as j:
                json.dump(ssim_results, j)



            with open(join(save_location, "psnr_results.json"), 'w') as j:
                json.dump(psnr_results, j)

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


def heatmaps(result_path):

    heatmap_metrics = [t for t in listdir(result_path) if t.endswith(".json")]

    def fix_decimals(value_list):
        new_list = []
        for d in value_list:
            new_list.append(round(d, 2))
        return new_list

    for hm in heatmap_metrics:
        with open(join(result_path, hm), 'r') as j:
            if hm.startswith("ssim"):
                ssim_results = json.load(j)
            else:
                psnr_results = json.load(j)

    # this makes sure the decimal range is constrained
    ssim_results["Poisson"] = fix_decimals(ssim_results["Poisson"])
    ssim_results["Gaussian"] = fix_decimals(ssim_results["Gaussian"])

    psnr_results["Poisson"] = fix_decimals(psnr_results["Poisson"])
    psnr_results["Gaussian"] = fix_decimals(psnr_results["Gaussian"])

    ssim_df = pd.DataFrame(ssim_results)
    psnr_df = pd.DataFrame(psnr_results)
    plt.rcParams['figure.figsize'] = 10, 8
    print(ssim_df.to_string())
    plt.figure()
    plt.title("SSIM Heatmap")
    ssim_data = prepare_2d_array_from_df(["Poisson", "Gaussian"], ssim_df)['Value']
    print(ssim_data)
    #print(ssim_df["Poisson"].unique().tolist())
    y_labels = np.sort(ssim_df["Poisson"].unique()).tolist()
    x_labels = np.sort(ssim_df["Gaussian"].unique()).tolist()
    ssim_heatmap = sns.heatmap(data=ssim_data, annot=True, yticklabels=y_labels, xticklabels=x_labels)
    ssim_heatmap.set_ylabel("Poisson")
    ssim_heatmap.set_xlabel("Gaussian")
    plt.tight_layout()
    plt.savefig(join(result_path, "SSIM.png"))
    plt.close()

    plt.rcParams['figure.figsize'] = 10, 8
    plt.figure()
    plt.title("PSNR Heatmap")
    psnr_data = prepare_2d_array_from_df(["Poisson", "Gaussian"], psnr_df)['Value']
    #print(ssim_df["Poisson"].unique().tolist())
    y_labels = np.sort(psnr_df["Poisson"].unique()).tolist()
    x_labels = np.sort(psnr_df["Gaussian"].unique()).tolist()
    psnr_heatmap = sns.heatmap(data=psnr_data, annot=True, yticklabels=y_labels, xticklabels=x_labels)
    psnr_heatmap.set_ylabel("Poisson")
    psnr_heatmap.set_xlabel("Gaussian")
    plt.tight_layout()
    plt.savefig(join(result_path, "PSNR.png"))


def recalculate_ssim(ref_im_path, noisy_im_path):
    '''
    This function recalculates the SSIM to be more in line with the paper which produced it. Due to this,
    the slices will need to be measured individually and then averaged as the z-axis constrains the gaussian
    weight approach
    :param ref_im_path:
    :param noisy_im_path:
    :return:
    '''
    noise_im_set = np.array([[f, f.split('p')[1].split('g')[0], f.split('p')[1].split('g')[1].split('.')[0]]
                             for f in listdir(noisy_im_path) if f.endswith('.tif')])
    ssim_results = {"Poisson": [], "Gaussian": [], "Value": []}
    ref_im = io.imread(ref_im_path)
    im_set_total = noise_im_set.shape[0]
    for t in range(noise_im_set.shape[0]):
        print("Image", t, "of", im_set_total)
        noise_im = io.imread(join(noisy_im_path, noise_im_set[t][0]))
        ssim_results["Poisson"].append(float(noise_im_set[t][1]))
        ssim_results["Gaussian"].append(float(noise_im_set[t][2]))
        stack_ssim = np.empty(noise_im.shape[0], dtype=float)
        for z in range(noise_im.shape[0]):
            stack_ssim[z] = structural_similarity(ref_im[z], noise_im[z], gaussian_weights=True, sigma=1.5,
                                                  use_sample_covariance=False, data_range=255)
        ssim_results["Value"].append(stack_ssim.mean())

    with open(join(noisy_im_path, "ssim_results.json"), 'w') as j:
        json.dump(ssim_results, j)


def apply_noise(blurred_path, dest_path, poiss_range, gauss_range):
    blurred_file_list = [f for f in listdir(blurred_path) if f.endswith('.tif')]

    for f in blurred_file_list:
        blurred_im = io.imread(join(blurred_path, f)).astype('uint64')
        sub_path = join(dest_path, f.split('.')[0])
        if not exists(sub_path):
            makedirs(sub_path)
        for p in poiss_range:
            for g in gauss_range:
                poisson_applied = apply_poisson(blurred_im, p)
                gauss_applied = apply_gaussian(poisson_applied, g)
                io.imsave(join(sub_path, f.split('.')[0] + "_p" + str(p) + "g" + str(g) + ".tif"), gauss_applied)


def test_poisson():
    blurred_im = io.imread(join("F:\\clean images\\Blurred\\", "CCCP_1C=1T=0.tif"))
    for s in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
        noise_im = apply_gaussian(apply_poisson(blurred_im.astype('uint8'), s), 1)
        io.imsave(join("F:\\clean images\\poisson_testing\\", "8bit_poiss_" + str(s) + "_.tif"), noise_im)
        noise_im = apply_gaussian(apply_poisson(blurred_im.astype('uint64'), s), 1)
        io.imsave(join("F:\\clean images\\poisson_testing\\", "64bit_poiss_" + str(s) + "_.tif"), noise_im)


if __name__ == "__main__":
    '''
    To apply Poisson and Gaussian noise to images in combinations:
    1. Provide the input and output paths where the images that noise is to be added to is in the input path
    2. Provide a range for the noise parameters for Poisson and Gaussian as lists of integers as shown below.
    3. It will create sub-folders for each sample at the destination which will contain the noise variations for each 
    sample.
    '''
    input_path = ""
    output_path = ""
    poisson_noise = [0.2, 1, 2, 3, 4]
    gaussian_noise = [1, 7, 14, 21, 28]
    apply_noise(input_path, output_path, poisson_noise, gaussian_noise)