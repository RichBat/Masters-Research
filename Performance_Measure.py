import numpy as np
import math

import pandas as pd
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage import data, io
from scipy import special, stats
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_sauvola, threshold_otsu, threshold_li
from skimage.filters.rank import otsu
from scipy import ndimage as ndi
from skimage import morphology
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, variation_of_information
import warnings
import cv2
import json
import mahotas
from mahotas.convolve import rank_filter
import seaborn as sns

from CleanThresholder import AutoThresholder

#py2pi has autothresholder but I don't know how to install

image_path = {"ref":"", "binary":"F:\\clean images\\Binarized\\Hysteresis\\"}

#Must split P for poisson files, PG and G for G+P and gaussian noise path.
thresh_image_path = "F:\\clean images\\Threshold_outputs\\"
'''To Do:
    1. Add function for stepping through the binary images and get the file names from that
    2. Load in the list of files for each of the noise paths and where necessary separate the file name from the
    noise application. Gaussian will have a split and under the file name will be a list of the gaussian var params to
    step through
    3. With this organised we can iterate through the images in the selected image path
    4. Also set-up a file iterator that can be limited (do three images only) to evaluate the gaussian severity
    5. Organise and save the results alongside the thresholded image
'''

data_structure = {"Sample":[], "Method":[], "Accuracy":[], "Precision":[], "Recall":[], "SNR":[], "SSIM":[]}

class PerformanceMeasuring(AutoThresholder):

    def __init__(self, input_paths):
        AutoThresholder.__init__(self, input_paths)
        self.threshold_result_holder = {"P":{}, "G":{}, "PG":{}}
        self.poisson_noise = None
        self.gaussian_noise = None
        self.pg_noise = None

    def _tuned_poisson(self, image, intensity):
        '''
        This is a duplicate of the Poisson mode noise in skimage.util.random_noise but with the added ability to
        increase the intensity of the poisson intensity without compounding multiple applications of Poisson onto
        the image.
        :param image:
        :param intensity:
        :return:
        '''
        if intensity == 0:
            return image
        rng = np.random.default_rng(None)
        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))

        # Ensure image is exclusively positive
        if low_clip == -1.:
            old_max = image.max()
            image = (image + 1.) / (old_max + 1.)

        # Generating noise for each unique value in image.
        og_poiss = rng.poisson(image * vals)

        # The major difference here is that instead of dividing by vals I normalize using the intensity max
        out = np.power(og_poiss / og_poiss.max(), intensity)
        out = out/out.max()

        return (out*image).astype('uint8')


    def _get_poisson_list(self, poisson_path):
        poisson_list = {f.split(sep="P.tif")[0]: join(poisson_path, f) for f in listdir(poisson_path) if
                        isfile(join(poisson_path, f)) and f.endswith(".tif")}
        return poisson_list

    def _get_gaussian_list(self, gaussian_path):
        gaussian_list = {}
        for g in listdir(gaussian_path):
            if isfile(join(gaussian_path, g)) and g.endswith(".tif"):
                g_split = g.split(sep="G")
                f_name, noise_var = g_split[0], g_split[1].split(sep=".tif")[0]
                if f_name not in gaussian_list:
                    gaussian_list[f_name] = {}
                gaussian_list[f_name][noise_var] = join(gaussian_path, g)

        return gaussian_list

    def _get_pgaussian_list(self, pgaussian_path):
        pgaussian_list = {}
        for g in listdir(pgaussian_path):
            if isfile(join(pgaussian_path, g)) and g.endswith(".tif"):
                g_split = g.split(sep="PG")
                f_name, noise_var = g_split[0], g_split[1].split(sep=".tif")[0]
                if f_name not in pgaussian_list:
                    pgaussian_list[f_name] = {}
                pgaussian_list[f_name][noise_var] = join(pgaussian_path, g)

        return pgaussian_list

    def apply_global_thresholds(self, image, grnd_truth, save_image=None):
        threshold_value = threshold_otsu(image)
        binary_output = np.greater(image, threshold_value).astype("uint8")*255
        otsu_results = self.metrics(binary_output, grnd_truth, save_image)
        return otsu_results

    def metrics(self, binary_image, ground_truth, save_image=None):
        if save_image is not None:
            io.imsave(save_image, binary_image.astype("uint8"))
        binary_image, ground_truth = np.clip(binary_image.astype(int), 0, 1), np.clip(ground_truth.astype(int), 0, 1)
        match = np.logical_not(np.logical_xor(binary_image, ground_truth)).astype(int)
        union = np.logical_and(binary_image, ground_truth).astype(int)
        '''plt.figure(1)
        io.imshow(np.amax(binary_image, axis=0))
        plt.figure(2)
        io.imshow(np.amax(ground_truth, axis=0))
        plt.show()'''
        accuracy = match.sum() / binary_image.size
        precision = union.sum() / binary_image.sum()
        recall = union.sum() / ground_truth.sum()

        return [accuracy, precision, recall]

    def apply_local_thresholds(self, image, grnd_truth, save_image=None):
        win_size = 15 # might put into argument or have as a range and iterate across?
        thresh_image = (image > threshold_sauvola(image, window_size=win_size)).astype("uint8")*255
        print("Sauvola complete")
        # save_image_path = thresh_image_path + "Sauvola\\" + save_image if save_image is not None else None
        sauvola_results = self.metrics(thresh_image, grnd_truth)
        window = morphology.cube(win_size)[0:1]
        thresh_image = (image > otsu(image, window)).astype("uint8")*255
        # save_image_path = thresh_image_path + "L_Otsu\\" + save_image if save_image is not None else None
        l_otsu_results = self.metrics(thresh_image, grnd_truth)
        print("Otsu complete")
        thresh_image = self.Bernsen_test(image, morphology.disk(15), 15).astype("uint8")*255
        # save_image_path = thresh_image_path + "Bernsen\\" + save_image if save_image is not None else None
        print("Bernsen")
        bernsen_results = self.metrics(thresh_image, grnd_truth)
        return sauvola_results, l_otsu_results, bernsen_results

    def Bernsen_test(self, image, neighbourhood, contr_thresh=15):
        # https://github.com/luispedro/mahotas/issues/84 used this to fix to be like FiJi
        from mahotas.convolve import rank_filter

        def apply_filter(flat_image):
            fmax = rank_filter(flat_image, neighbourhood, neighbourhood.sum() - 1)
            fmin = rank_filter(flat_image, neighbourhood, 0)
            fptp = fmax - fmin
            fmean = fmax / 2. + fmin / 2.  # Do not use (fmax + fmin) as that may overflow
            return np.choose(fptp < contr_thresh, (flat_image >= fmean, flat_image >= 128))
        if len(image.shape) > 2:
            output_image = np.zeros_like(image)
            for z in range(image.shape[0]):
                output_image[z] = apply_filter(image[z])
            return output_image
        else:
            return apply_filter(image)


    def ihh_threshold(self, image, grnd_truth, save_image=None):
        gray_image = self._grayscale(image)
        idhb = self._specific_thresholds(gray_image, ["Logistic"], [0, 1, 2])
        if idhb is None:
            print("None for this")
        else:
            option_results = {}
            for t in idhb["Inverted"]:
                opt_threshes = idhb["Inverted"][t]
                thresh_image = self._threshold_image(gray_image, opt_threshes[0], opt_threshes[1]).astype("uint8")*255
                '''plt.figure(1)
                io.imshow(thresh_image[2])
                plt.figure(2)
                io.imshow(grnd_truth[2])
                plt.show()'''
                print(save_image)
                save_w_option = thresh_image_path + "Inverted\\" + save_image + "Opt" + str(t) + ".tif"
                option_results[t] = self.metrics(thresh_image, grnd_truth, save_w_option)
            return option_results


    def threshold_samples(self, limit=0):
        counter = 0
        for f in self.file_list:
            if limit != 0:
                counter += 1
                if counter > limit:
                    return
            grnd_truth = io.imread(f[0])
            file_name = f[1].split(".tif")[0]
            print(file_name)
            # Image Derived Hysteresis Binarization

            thresh_image = io.imread(self.poisson_noise[file_name])
            poisson_global = self.apply_global_thresholds(thresh_image, grnd_truth, file_name+"P.tif")
            print("Global done")
            poisson_sauvola, poisson_otsu, p_bernsen = self.apply_local_thresholds(thresh_image, grnd_truth, file_name+"P.tif")
            print("Local Done")
            idhb_thresholds = self.ihh_threshold(thresh_image, grnd_truth, file_name+"P")
            print("IHH Done")
            self.threshold_result_holder["P"][f[0]] = {"Otsu":poisson_global, "Sauvola":poisson_sauvola,
                                                       "L_Otsu":poisson_otsu, "Bernsen":p_bernsen,
                                                       "IHH":idhb_thresholds}

            print("Poisson done")
            gaussian_vars = list(self.gaussian_noise[file_name].keys())
            for g in gaussian_vars:
                print("Var " + str(g))
                if g not in list(self.threshold_result_holder["G"]):
                    self.threshold_result_holder["G"][g] = {}
                thresh_image = io.imread(self.gaussian_noise[file_name][g])
                gauss_global = self.apply_global_thresholds(thresh_image, grnd_truth, file_name
                                                            +"G"+g+".tif")
                print("Gaussian Global done")
                gauss_sauvola, gauss_otsu, g_bernsen = self.apply_local_thresholds(thresh_image, grnd_truth, file_name
                                                            +"G"+g+".tif")
                print("Gaussian Local done")
                idhb_thresholds = self.ihh_threshold(thresh_image, grnd_truth, file_name
                                                            +"G"+g)
                print("IHH Done")
                self.threshold_result_holder["G"][g][f[0]] = {"Otsu": gauss_global, "Sauvola": gauss_sauvola,
                                                           "L_Otsu": gauss_otsu, "Bernsen": g_bernsen,
                                                              "IHH": idhb_thresholds}
            print("Gaussian Done")
            pgauss_vars = list(self.pg_noise[file_name].keys())
            for pg in pgauss_vars:
                if pg not in list(self.threshold_result_holder["PG"]):
                    self.threshold_result_holder["PG"][pg] = {}
                thresh_image = io.imread(self.pg_noise[file_name][g])
                pg_global = self.apply_global_thresholds(thresh_image, grnd_truth, file_name
                                                            +"PG"+pg+".tif")
                pg_sauvola, pg_otsu, pg_bernsen = self.apply_local_thresholds(thresh_image, grnd_truth, file_name
                                                            +"PG"+pg+".tif")
                idhb_thresholds = self.ihh_threshold(thresh_image, grnd_truth, file_name
                                                            +"PG"+pg)
                self.threshold_result_holder["PG"][pg][f[0]] = {"Otsu": pg_global, "Sauvola": pg_sauvola,
                                                              "L_Otsu": pg_otsu, "Bernsen":pg_bernsen,
                                                                "IHH":idhb_thresholds}
            print("P Gaussian done")


    def fix_missing_images(self):
        '''
        This method will fix the IHH images being misnamed and the missing Otsu Poisson images
        :return:
        '''
        for f in self.file_list:
            grnd_truth = io.imread(f[0])
            file_name = f[1].split(".tif")[0]
            gaussian_vars = list(self.gaussian_noise[file_name].keys())
            for g in gaussian_vars:
                print("Var " + str(g))
                thresh_image = io.imread(self.gaussian_noise[file_name][g])
                idhb_thresholds = self.ihh_threshold(thresh_image, grnd_truth, file_name
                                                     + "G" + g)
            pgauss_vars = list(self.pg_noise[file_name].keys())
            for pg in pgauss_vars:
                thresh_image = io.imread(self.pg_noise[file_name][g])
                idhb_thresholds = self.ihh_threshold(thresh_image, grnd_truth, file_name
                                                     + "PG" + pg)
            # Image Derived Hysteresis Binarization
            thresh_image = io.imread(self.poisson_noise[file_name])
            poisson_global = self.apply_global_thresholds(thresh_image, grnd_truth, file_name+"P.tif")

    def save_threshold_results(self, save_path):
        with open(save_path, 'w') as j:
            json.dump(self.threshold_result_holder, j)

    def test_poisson(self):
        for f in self.file_list:
            image = io.imread(f[0])
            mip = np.amax(image, axis=0)
            poisson_image = self._tuned_poisson(mip, 1)
            poisson_image2 = self._tuned_poisson(mip, 10)

            fig1 = plt.figure()
            io.imshow(poisson_image)
            fig2 = plt.figure()
            io.imshow(poisson_image2)
            fig3 = plt.figure()
            io.imshow(random_noise(mip, mode='poisson'))
            plt.show()

    def _iterate_noise(self, save_location, save_im=True):

        noise_metrics = {"Sample":[], "Poisson":[], "Gaussian":[], "SNR":[], "SSIM":[]}

        def _add_to_noise_metrics(sample_name, p_value, g_value, snr, ssim):
            noise_metrics["Sample"].append(sample_name)
            noise_metrics["Poisson"].append(p_value)
            noise_metrics["Gaussian"].append(g_value)
            noise_metrics["SNR"].append(snr)
            noise_metrics["SSIM"].append(ssim)

        def _apply_gaussian(im, intensity):
            if intensity == 0:
                return im.astype('uint8')
            else:
                return (random_noise(image=im, var=intensity) * 255).astype('uint8')

        for f in self.file_list:
            file_name = f[1].split('.tif')[0]
            image = io.imread(f[0])
            new_path = join(save_location, file_name)
            if not exists(new_path):
                makedirs(new_path)

            for p in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for g in [0, 0.0003, 0.0006, 0.0009, 0.0012, 0.0015, 0.0018, 0.0021]:
                    noisy_sample_name = file_name + "P" + str(p) + "G" + str(g) + ".tif"
                    poisson_application = self._tuned_poisson(image, p)
                    gaussian_application = _apply_gaussian(poisson_application, g)

                    snr_value = peak_signal_noise_ratio(image, gaussian_application)
                    window_size = image.shape[0] if image.shape[0] % 2 != 0 else image.shape[0] - 1
                    ssim_value = structural_similarity(image, gaussian_application, win_size=window_size)

                    _add_to_noise_metrics(file_name, p, g, snr_value, ssim_value)
                    if save_im:
                        io.imsave(join(new_path, noisy_sample_name), gaussian_application)

            with open(join(save_location, "Noise_metrics.json"), 'w') as j:
                json.dump(noise_metrics, j)


            #here is where noise will be iteratively applied with a dictionary to save the noise ratios for graphing
            #with the SNR and maybe SSIM


    def _get_metrics(self, ground_truth, image, storage_structure):
        window_size = image.shape[0] if image.shape[0] % 2 != 0 else image.shape[0] - 1
        snr = peak_signal_noise_ratio(ground_truth.astype('uint8'), image.astype('uint8'))
        ssim = structural_similarity(ground_truth.astype('uint8'), image.astype('uint8'), win_size=window_size)
        pixel_metrics = self.metrics(image, ground_truth)
        accuracy, precision, recall = pixel_metrics[0], pixel_metrics[1], pixel_metrics[2]
        storage_structure["Accuracy"].append(accuracy)
        storage_structure["Precision"].append(precision)
        storage_structure["Recall"].append(recall)
        storage_structure["SNR"].append(snr)
        storage_structure["SSIM"].append(ssim)


    def _local_thresh(self, method, image, window_size):
        '''
        This method will get the local threshold for 1 of 3 methods. The window_size is percentage based on the
        smaller dimension of the image shape (excluding the Z-axis). This way this can be varied and reported on
        dimensions of the
        :param method:
        :param image:
        :param window_size:
        :return:
        '''
        '''win_size = int(min(image.shape[1:]) * (window_size/100))
        win_size = win_size + 1 if win_size % 2 == 0 else win_size'''
        win_size = window_size
        #print("Window size", win_size, type(win_size))
        if method == "Sauvola":
            thresh_image = (image > threshold_sauvola(image, window_size=win_size)).astype("uint8")
        if method == "Otsu":
            thresh_image = (image > otsu(image, morphology.cube(win_size)[0:1])).astype("uint8")
        if method == "Bernsen":
            thresh_image = self.Bernsen_test(image, morphology.disk(win_size), 15).astype("uint8")
        return thresh_image


    def apply_thresholds(self, image, ground_truth, sample_name, g, p, storage_array, global_thresh=False, local_thresh=False):
        if global_thresh:

            otsu_results = np.greater(image, threshold_otsu(image)).astype("uint8")*ground_truth
            self._get_metrics(ground_truth, otsu_results, storage_array)
            storage_array["Sample"].append(sample_name)
            storage_array["Method"].append("Otsu")
            storage_array["G"].append(g)
            storage_array["P"].append(p)
            li_results = np.greater(image, threshold_li(image)).astype("uint8")*ground_truth
            self._get_metrics(ground_truth, li_results, storage_array)
            storage_array["Sample"].append(sample_name)
            storage_array["Method"].append("Li")
            storage_array["G"].append(g)
            storage_array["P"].append(p)
        if local_thresh:

            for w in [15]:
                print("Starting Sauvola")
                storage_array["Sample"].append(sample_name)
                storage_array["Method"].append("Sauvola")
                storage_array["Window"].append(w)
                storage_array["G"].append(g)
                storage_array["P"].append(p)
                thresh_image = self._local_thresh("Sauvola", image, w)*ground_truth
                self._get_metrics(ground_truth, thresh_image, storage_array)
                print("Starting Otsu")
                storage_array["Sample"].append(sample_name)
                storage_array["Method"].append("Otsu")
                storage_array["Window"].append(w)
                storage_array["G"].append(g)
                storage_array["P"].append(p)
                thresh_image = self._local_thresh("Otsu", image, w)*ground_truth
                self._get_metrics(ground_truth, thresh_image, storage_array)




    def iterate_thresholding(self, noise_path):
        '''
        This function will do global thresholding for thresh_type==0 and local for thresh_type==1
        :param noise_path:
        :param thresh_type:
        :return:
        '''
        global_data = {"Sample": [], "Method": [], "G": [], "P": [], "Accuracy": [], "Precision": [],
                       "Recall": [],
                       "SNR": [], "SSIM": []}
        local_data = {"Sample": [], "Method": [], "Window": [], "G": [], "P": [], "Accuracy": [],
                      "Precision": [],
                      "Recall": [], "SNR": [], "SSIM": []}
        samples_covered = [[], []]

        '''if isfile(join(noise_path, "global_thresh.json")):
            with open(join(noise_path, "global_thresh.json"), 'r') as j:
                global_data = json.load(j)
            global_df = pd.DataFrame(global_data)
            samples_covered[0] = pd.unique(global_df['Sample']).to_list()'''

        if isfile(join(noise_path, "local_thresh.json")):
            with open(join(noise_path, "local_thresh.json"), 'r') as j:
                local_data = json.load(j)
            local_df = pd.DataFrame(local_data)
            samples_covered[1] = pd.unique(local_df['Sample']).tolist()

        for t in range(0, len(self.file_list)):
            sample_name = self.file_list[t][1].split('.')[0]
            print(sample_name, "***********", (t + 1), '/', len(self.file_list))
            if sample_name not in samples_covered[1]:
                ground_truth = io.imread(self.file_list[t][0])
                sample_noise_path = join(noise_path, sample_name)
                noise_file_list = [nf[:-4] for nf in listdir(sample_noise_path)]
                noise_count = len(noise_file_list)
                noise_counter = 0
                for noise_file in noise_file_list:
                    noise_counter += 1
                    print("Noise variant:", noise_counter, '/', noise_count)
                    noise_params = noise_file.split("C=")[1].split("P")[1].split("G")
                    poisson_intens = noise_params[0]
                    gauss_intens = noise_params[1]
                    noise_image = io.imread(join(sample_noise_path, noise_file+".tif"))
                    #Global Threshold
                    '''self.apply_thresholds(noise_image, ground_truth, sample_name, gauss_intens, poisson_intens, global_data,
                                          global_thresh=True)'''
                    #Local Threshold
                    self.apply_thresholds(noise_image, ground_truth, sample_name, gauss_intens, poisson_intens, local_data,
                                          local_thresh=True)

                '''with open(join(noise_path, "global_thresh.json"), 'w') as j:
                    json.dump(global_data, j)'''
                with open(join(noise_path, "local_thresh.json"), 'w') as j:
                    json.dump(local_data, j)
            else:
                print("Sample already present!!!")

    def ihh_iterate(self, noise_path, j_name=None):
        samples_covered = []
        if j_name is None:
            ihh_results = {"Sample": [], "Window": [], "Voxel":[], "G": [], "P": [], "Accuracy": [],
                          "Precision": [], "Recall": [], "SNR": [], "SSIM": []}
        else:
            with open(join(noise_path, j_name), 'w') as j:
                ihh_results = json.load(j)
            data_check = pd.DataFrame(ihh_results)
            samples_covered = pd.unique(data_check['Sample']).to_list()



        def record_other(sample_name, win_opt, voxel_bias, g, p):
            ihh_results["Sample"].append(sample_name)
            ihh_results["Window"].append(win_opt)
            ihh_results["Voxel"].append(voxel_bias)
            ihh_results["G"].append(g)
            ihh_results["P"].append(p)

        for t in range(0, len(self.file_list)):
            sample_name = self.file_list[t][1].split('.')[0]
            print(sample_name, "***********", (t + 1), '/', len(self.file_list))
            if sample_name not in samples_covered:
                ground_truth = io.imread(self.file_list[t][0])
                sample_noise_path = join(noise_path, sample_name)
                noise_file_list = [nf[:-4] for nf in listdir(sample_noise_path)]
                noise_counter = 0
                for noise_file in noise_file_list:
                    noise_counter += 1
                    print(noise_file)
                    print("Noise file", noise_counter, " of", len(noise_file_list))
                    noise_params = noise_file.split("C=")[1].split("P")[1].split("G")
                    poisson_intens = noise_params[0]
                    gauss_intens = noise_params[1]
                    noise_image = io.imread(join(sample_noise_path, noise_file + ".tif"))

                    #No voxel bias
                    vb = False
                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 0)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(int)*ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 1, vb, gauss_intens, poisson_intens)

                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 1)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(int)*ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 2, vb, gauss_intens, poisson_intens)

                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 2)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(int)*ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 0, vb, gauss_intens, poisson_intens)

                    #With Voxel Bias
                    vb = True
                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 0)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(
                        int) * ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 1, vb, gauss_intens, poisson_intens)

                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 1)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(
                        int) * ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 2, vb, gauss_intens, poisson_intens)

                    high_thresh, low_thresh = self.inverted_thresholding_final(noise_image, vb, 2)
                    thresh_im = apply_hysteresis_threshold(noise_image, low_thresh, high_thresh).astype(
                        int) * ground_truth
                    self._get_metrics(ground_truth, thresh_im, ihh_results)
                    record_other(sample_name, 0, vb, gauss_intens, poisson_intens)

                    with open(join(noise_path, "ihh_results.json"), 'w') as j:
                        json.dump(ihh_results, j)

    def test_json_reading(self, test_path):
        with open(test_path, 'r') as j:
            test_data = json.load(j)
        test_df = pd.DataFrame(test_data)
        print(type(pd.unique(test_df['Sample'])))
        list_convert = pd.unique(test_df['Sample'])
        print(list_convert)
        print("CCCP_1C=1T=0" in list_convert)

    def test_ihh_control_performance(self, noise_path, noise_subsets):
        '''
        The noise path will be the path to the noise sample sub-folder parent (will be the folder that contains
        the folders for the noise variations of each sample). The noise subsets will be two lists in a tuple
        with the first being the Poisson and the second being Gaussian. The permutations of these will be used
        to get a limited subset of the noise outputs.
        :param noise_path:
        :param noise_subsets:
        :return:
        '''
        poison_noise_options = noise_subsets[0]
        gaussian_noise_options = noise_subsets[1]
        for t in range(0, len(self.file_list)):
            sample_name = self.file_list[t][1].split('.')[0]
            print(sample_name, "***********", (t + 1), '/', len(self.file_list))
            sample_noise_path = join(noise_path, sample_name)
            for p in poison_noise_options:
                for g in gaussian_noise_options:
                    noise_file = sample_name + "P" + str(p) + "G" + str(g) + '.tif'
                    noise_sample_path = join(sample_noise_path, noise_file)
                    noise_im = io.imread(noise_sample_path)
                    low_thresh, valid_low = self._low_select(noise_im)
                    intens, voxels = self._ihh_get_best(noise_im, low_thresh, test_distrib=True)
                    slopes, slope_points = self._get_slope(intens, voxels)
                    mving_slopes = self._moving_average(slopes, window_size=8)
                    inverted_slopes, inversion_record = self._invert_rescaler(mving_slopes)
                    voxel_weights = voxels / voxels.max()
                    print("Noise combo", p, g)
                    print("Low thresh value", low_thresh)
                    #normal centroid
                    high1, low_thresh = self.inverted_thresholding_final(noise_im, voxel_bias=False, window_option=0,
                                                                         testing_ihh=False)
                    #Full distrib centroid
                    high2, low_thresh = self.inverted_thresholding_final(noise_im, voxel_bias=False, window_option=0,
                                                                         testing_ihh=True)

                    sns.lineplot(x=intens[:-1], y=inverted_slopes, color='r')
                    sns.lineplot(x=intens, y=voxel_weights, color='g')
                    plt.axvline(x=high1, color='k')
                    plt.axvline(x=high2, color='b')
                    plt.show()


if __name__ == "__main__":
    noise_metrics = PerformanceMeasuring([image_path["binary"]])
    #noise_metrics.test_json_reading("F:\\clean images\\Noise Applied\\Noise_metrics.json")
    #noise_metrics.ihh_iterate("F:\\clean images\\Noise Applied\\")
    #noise_metrics.test_ihh_control_performance("F:\\clean images\\Noise Applied\\", ([0, 4, 10], [0, 0.0012, 0.0021]))
    #noise_metrics.iterate_thresholding("F:\\clean images\\Noise Applied\\")
    #noise_metrics._iterate_noise("F:\\clean images\\Noise Applied\\", True)
    #noise_metrics.test_poisson()
    #noise_metrics.fix_missing_images()
    #noise_metrics.threshold_samples()
    #noise_metrics.save_threshold_results("C:\\Users\\richy\\Desktop\\clean images\\metric_results.json")