import numpy as np
import math
from skimage.exposure import histogram
from skimage import data, io, util
from scipy import special, stats
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage import morphology
import warnings
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "C:\\Users\\richy\\Desktop\\clean images\\Blurred\\"

blurred_files = [(join(file_path, f), f) for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith(".tif")]
noise_path = "C:\\Users\\richy\\Desktop\\clean images\\Noise Applied\\"
noise_folders = ["G+P\\", "Gaussian\\", "Poisson\\"]

def iterate_files():
    for bf in blurred_files:
        blurred_image = io.imread(bf[0])
        filename = bf[1].split(sep='.')[0]
        add_noise(blurred_image, filename)


def add_noise(image, file_name):
    var = [0.0001, 0.0005, 0.0007, 0.01]
    poisson_noise = util.random_noise(image, mode='poisson')
    io.imsave(noise_path + noise_folders[2] + file_name + 'P.tif', (poisson_noise * 255).astype('uint8'))
    gaussian_noise = (util.random_noise(image, var=var[0]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[1] + file_name + "G" + str(var[0]) + '.tif', gaussian_noise)
    gaussian_noise1 = (util.random_noise(image, var=var[1]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[1] + file_name + "G" + str(var[1]) + '.tif', gaussian_noise1)
    gaussian_noise2 = (util.random_noise(image, var=var[2])*255).astype('uint8')
    io.imsave(noise_path + noise_folders[1] + file_name + "G" + str(var[2]) + '.tif', gaussian_noise2)
    gaussian_noise3 = (util.random_noise(image, var=var[3]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[1] + file_name + "G" + str(var[3]) + '.tif', gaussian_noise3)

    p_gaussian_noise = (util.random_noise(image, var=var[0]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[0] + file_name + "PG" + str(var[0]) + '.tif', p_gaussian_noise)
    p_gaussian_noise1 = (util.random_noise(image, var=var[1]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[0] + file_name + "PG" + str(var[1]) + '.tif', p_gaussian_noise1)
    p_gaussian_noise2 = (util.random_noise(image, var=var[2]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[0] + file_name + "PG" + str(var[2]) + '.tif', p_gaussian_noise2)
    p_gaussian_noise3 = (util.random_noise(image, var=var[3]) * 255).astype('uint8')
    io.imsave(noise_path + noise_folders[0] + file_name + "PG" + str(var[3]) + '.tif', p_gaussian_noise3)



if __name__ == "__main__":
    iterate_files()