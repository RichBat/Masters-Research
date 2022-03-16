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

input_path = "C:\\RESEARCH\\Mitophagy_data\\Fixed_Images\\"
source_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
substring = "Compared"
precision = 10
density = 30

images = [f for f in listdir(input_path) if isfile(join(input_path, f)) and substring in f]
#separated_images = [[image.split('density')[0], image.split('density')[1].split('resolution')[0], ] for image in images]
specific_samples = [[input_path + image, source_path + image.split('density')[0] + '.tif'] for image in images if int(image.split('density')[1].split('resolution')[0]) == density and
                    float(image.split('density')[1].split('resolution')[1].split('highThresh')[0]) == precision]

otsu_greater_list = []
otsu_cutoff = 0.001
noise_ra = 0.05

class ThresholdOutlierDetector:

    def __init__(self, source_image_path, sample_name):
        self.otsu_greater_list = []
        self.source_image = io.imread(source_image_path + sample_name)
        self.noise = self.noiseGeneration()
        self.noisy_source = io.imread(source_image_path + sample_name)

    def noiseGeneration(self):
        noisy = random_noise(image=self.source_image, mode='gaussian')
        noise = (noisy - self.source_image / np.max(self.source_image))
        return noise

    def generate_noisy_image(self, noise_ratio):
        source_max = np.max(self.source_image)
        self.noisy_source = ((self.source_image/source_max)+self.noise*noise_ratio)*source_max
        self.noisy_source[np.where(self.noisy_source < 0)] = 0
        self.noisy_source = np.round(self.noisy_source).astype("int")
        return self.noisy_source

    def outlierDetection(self, thresholded_image, otsu_cutoff=0.001):
        otsu_threshold = threshold_otsu(self.noisy_source, 256)
        otsu_image = np.zeros_like(self.noisy_source)
        otsu_image[np.where(self.noisy_source >= otsu_threshold)] = 1
        otsu_voxels = otsu_image.sum()*otsu_cutoff
        threshold_voxels = np.round(thresholded_image/np.max(thresholded_image)).sum()
        return threshold_voxels > otsu_voxels, threshold_voxels, otsu_voxels

def testrun():
    for sample in specific_samples:
        compared_image = io.imread(sample[0])
        source_image = io.imread(sample[1])
        blurred_source = np.zeros_like(source_image)
        for src in range(source_image.shape[0]):
            blurred_source[src] = cv.GaussianBlur(source_image[src], (0,0), 10)
        noisy = random_noise(image=source_image, mode='gaussian')
        noise = (noisy - source_image/np.max(source_image))*noise_ra
        snr = np.mean(source_image/np.max(source_image))/np.std(noise)
        print("SNR:", snr)
        io.imshow(source_image[2])
        plt.show()
        source_image = blurred_source
        noisy_source = source_image/np.max(source_image) + noise
        #io.imshow(source_image[2])
        #plt.show()
        #io.imshow(noisy_source[2]*np.max(source_image))
        #plt.show()
        print(np.mean(source_image))
        counts, centers = histogram(source_image)
        source_mean = np.mean(source_image)
        new_center = np.greater_equal(centers, source_mean).astype(int)
        counts = counts * new_center
        #source_image[np.where(source_image < source_mean)] = 0
        rensu_image = (np.zeros_like(compared_image[:,:,:,0]) + compared_image[:,:,:,0])
        rensu_image = rensu_image/np.max(rensu_image)
        richard_image = (np.zeros_like(compared_image[:,:,:,0]) + compared_image[:,:,:,1])
        richard_image = richard_image/np.max(richard_image)
        auto_image = (np.zeros_like(compared_image[:,:,:,0]) + compared_image[:,:,:,2])
        auto_image = auto_image/np.max(auto_image)
        otsu_thresh = threshold_otsu(source_image, 256)
        otsu_image = np.zeros_like(source_image)
        print("Otsu Thresh", otsu_thresh)
        otsu_image[np.where(source_image >= otsu_thresh)] = 1
        io.imshow(otsu_image[2])
        plt.show()
        #thr, second_otsu_image = cv.threshold(source_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #print(second_otsu_image.shape)
        print("Voxels Thresholded for " + sample[1].split(source_path)[1] + "\n" + str(rensu_image.sum()) + " " +
              str(richard_image.sum()) + " " + str(auto_image.sum()) + " " + str(otsu_image.sum()*otsu_cutoff) + "\n")
        print("Otsu greater: ", (otsu_image.sum()*otsu_cutoff > rensu_image.sum()), (otsu_image.sum()*otsu_cutoff > richard_image.sum()), (otsu_image.sum()*otsu_cutoff > auto_image.sum()))
        if otsu_image.sum()*otsu_cutoff > auto_image.sum():
            otsu_greater_list.append(sample[1].split(source_path)[1])

    print("Samples less than Otsu\n", otsu_greater_list)

if __name__ == "__main__":
    testrun()