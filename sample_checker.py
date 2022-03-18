import numpy
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu, threshold_otsu
from skimage import data, io
from skimage.exposure import histogram
from skimage.util import random_noise
import numpy as np
from os import listdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.util import img_as_float


input_path = "C:\\RESEARCH\\Mitophagy_data\\Fixed_Images\\"
source_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
substring = "Compared"
precision = 10
density = 30

otsu_greater_list = []
otsu_cutoff = 0.001
noise_ra = 0.05

class ThresholdOutlierDetector:

    def __init__(self, source_image_path, sample_name, noise_type='gaussian'):
        self.otsu_greater_list = []
        self.source_image = io.imread(source_image_path + sample_name)
        self.noise = self.noiseGeneration(noise_type)
        self.noisy_source = io.imread(source_image_path + sample_name)

    def noiseGeneration(self, noise_type):
        noisy = random_noise(image=self.source_image, mode=noise_type)
        float_img = img_as_float(self.source_image)
        noise2 = np.random.default_rng(None).normal(0, 0.02, float_img.shape)*np.max(self.source_image)
        noisy_float = float_img + noise2
        #clipped_float = np.clip(noisy_float, 0, np.max(self.source_image))
        noise = (noisy_float - float_img)
        #print(np.max(float_img), np.max(noise2), np.min(noise2))
        return noise

    def generate_noisy_image(self, noise_ratio):
        source_max = np.max(self.source_image)
        self.noisy_source = self.source_image+self.noise*noise_ratio
        self.noisy_source = np.clip(self.noisy_source, 0, np.max(self.source_image))
        self.noisy_source = np.round(self.noisy_source).astype("int")
        return self.noisy_source

    def outlierDetection(self, thresholded_image, otsu_cutoff=0.001):
        otsu_threshold = threshold_otsu(self.noisy_source, 256)
        otsu_image = np.zeros_like(self.noisy_source)
        otsu_image[np.where(self.noisy_source >= otsu_threshold)] = 1
        otsu_voxels = otsu_image.sum()*otsu_cutoff
        threshold_voxels = int(np.round(thresholded_image/np.max(thresholded_image)).sum())
        return threshold_voxels > otsu_voxels, threshold_voxels, otsu_voxels

def add_noise_to_deconvolved():
    deconvolved_image_path = "C:\\RESEARCH\Mitophagy_data\\Threshold Test Data\\Input Data postDeconvolution\\"
    noisy_save_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Input Data with Noise\\"
    images = [f for f in listdir(deconvolved_image_path) if isfile(join(deconvolved_image_path, f))]
    for img in images:
        sample_noise = ThresholdOutlierDetector(deconvolved_image_path, img)
        for noise in range(0, 101, 25):
            noisy_sample = sample_noise.generate_noisy_image(noise/100).astype('uint8')
            sampleName = img.split('.')[0] + "Noise" + str(noise) + ".tif"
            io.imsave(noisy_save_path+sampleName, noisy_sample)

if __name__ == "__main__":
    add_noise_to_deconvolved()