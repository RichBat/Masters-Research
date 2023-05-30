import numpy as np
import math
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage import data, io
from scipy import special, stats
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_sauvola, threshold_otsu
from skimage.filters.rank import otsu
from scipy import ndimage as ndi
from skimage import morphology
import warnings
import cv2
import json
import mahotas
from mahotas.convolve import rank_filter

from CleanThresholder import AutoThresholder

#py2pi has autothresholder but I don't know how to install

image_path = {"ref":"", "binary":"C:\\Users\\richy\\Desktop\\clean images\\Binarized\\Hysteresis\\"}
added__noise_path = ["C:\\Users\\richy\\Desktop\\clean images\\Noise Applied\\G+P\\",
                     "C:\\Users\\richy\\Desktop\\clean images\\Noise Applied\\Gaussian\\",
                     "C:\\Users\\richy\\Desktop\\clean images\\Noise Applied\\Poisson\\"
                     ] # This will have an integer for the percentage of added noise and the folder layer path
#Must split P for poisson files, PG and G for G+P and gaussian noise path.
thresh_image_path = "C:\\Users\\richy\\Desktop\\clean images\\Threshold_outputs\\"
'''To Do:
    1. Add function for stepping through the binary images and get the file names from that
    2. Load in the list of files for each of the noise paths and where necessary separate the file name from the
    noise application. Gaussian will have a split and under the file name will be a list of the gaussian var params to
    step through
    3. With this organised we can iterate through the images in the selected image path
    4. Also set-up a file iterator that can be limited (do three images only) to evaluate the gaussian severity
    5. Organise and save the results alongside the thresholded image
'''

class PerformanceMeasuring(AutoThresholder):

    def __init__(self, input_paths):
        AutoThresholder.__init__(self, input_paths)
        self.threshold_result_holder = {"P":{}, "G":{}, "PG":{}}
        self.poisson_noise = self._get_poisson_list(added__noise_path[2])
        self.gaussian_noise = self._get_gaussian_list(added__noise_path[1])
        self.pg_noise = self._get_pgaussian_list(added__noise_path[0])

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
        save_image_path = thresh_image_path+"Otsu\\"+save_image if save_image is not None else None
        otsu_results = self.metrics(binary_output, grnd_truth, save_image_path)
        #Huang or something else will go here
        return otsu_results

    def metrics(self, binary_image, ground_truth, save_image=None):
        if save_image is not None:
            io.imsave(save_image, binary_image.astype("uint8"))
        binary_image, ground_truth = np.clip(binary_image.astype(int), 0, 1), np.clip(ground_truth.astype(int), 0, 1)
        union = np.logical_and(binary_image, ground_truth).astype(int)
        accuracy = union.sum() / binary_image.size
        precision = union.sum() / binary_image.sum()
        recall = union.sum() / ground_truth.sum()
        return [accuracy, precision, recall]

    def apply_local_thresholds(self, image, grnd_truth, save_image=None):
        win_size = 15 # might put into argument or have as a range and iterate across?
        thresh_image = (image > threshold_sauvola(image, window_size=win_size)).astype("uint8")*255
        print("Sauvola complete")
        save_image_path = thresh_image_path + "Sauvola\\" + save_image if save_image is not None else None
        sauvola_results = self.metrics(thresh_image, grnd_truth, save_image_path)
        window = morphology.cube(win_size)[0:1]
        thresh_image = (image > otsu(image, window)).astype("uint8")*255
        save_image_path = thresh_image_path + "L_Otsu\\" + save_image if save_image is not None else None
        l_otsu_results = self.metrics(thresh_image, grnd_truth, save_image_path)
        print("Otsu complete")
        thresh_image = self.Bernsen_test(image[3], morphology.disk(15), 15)
        save_image_path = thresh_image_path + "Bernsen\\" + save_image if save_image is not None else None
        bernsen_results = self.metrics(thresh_image, grnd_truth, save_image_path)
        return sauvola_results, l_otsu_results, bernsen_results

    def Bernsen_test(self, image, neighbourhood, contr_thresh=15):
        # https://github.com/luispedro/mahotas/issues/84 used this to fix to be like FiJi
        from mahotas.convolve import rank_filter
        fmax = rank_filter(image, neighbourhood, neighbourhood.sum() - 1)
        fmin = rank_filter(image, neighbourhood, 0)
        fptp = fmax - fmin
        fmean = fmax / 2. + fmin / 2.  # Do not use (fmax + fmin) as that may overflow
        return np.choose(fptp < contr_thresh, (image >= fmean, image >= 128))

    def ihh_threshold(self, image):
        gray_image = self._grayscale(image)


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

            thresh_image = io.imread(self.poisson_noise[file_name])
            poisson_global = self.apply_global_thresholds(thresh_image, grnd_truth)
            print("Global done")
            poisson_sauvola, poisson_otsu = self.apply_local_thresholds(thresh_image, grnd_truth, file_name+"P.tif")
            print("Local Done")
            self.threshold_result_holder["P"][f[0]] = {"Otsu":poisson_global, "Sauvola":poisson_sauvola,
                                                       "L_Otsu":poisson_otsu}
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
                gauss_sauvola, gauss_otsu = self.apply_local_thresholds(thresh_image, grnd_truth, file_name
                                                            +"G"+g+".tif")
                self.threshold_result_holder["G"][g][f[0]] = {"Otsu": gauss_global, "Sauvola": gauss_sauvola,
                                                           "L_Otsu": gauss_otsu}
            print("Gaussian Done")
            pgauss_vars = list(self.pg_noise[file_name].keys())
            for pg in pgauss_vars:
                if pg not in list(self.threshold_result_holder["PG"]):
                    self.threshold_result_holder["PG"][pg] = {}
                thresh_image = io.imread(self.pg_noise[file_name][g])
                pg_global = self.apply_global_thresholds(thresh_image, grnd_truth, file_name
                                                            +"PG"+pg+".tif")
                pg_sauvola, pg_otsu = self.apply_local_thresholds(thresh_image, grnd_truth, file_name
                                                            +"PG"+pg+".tif")
                self.threshold_result_holder["PG"][pg][f[0]] = {"Otsu": pg_global, "Sauvola": pg_sauvola,
                                                              "L_Otsu": pg_otsu}
            print("P Gaussian done")
    def save_threshold_results(self, save_path):
        with open(save_path, 'w') as j:
            json.dump(self.threshold_result_holder, j)


if __name__ == "__main__":
    noise_metrics = PerformanceMeasuring([image_path["binary"]])
    noise_metrics.threshold_samples(limit=1)
    noise_metrics.save_threshold_results("C:\\Users\\richy\\Desktop\\clean images\\metric_results.json")