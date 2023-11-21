import numpy as np
from skimage import data, io
from os import listdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt

input_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
output_path = "C:\\RESEARCH\\Mitophagy_data\\Fixed_Images\\"
images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
compare_images = [i for i in images if "Compared" in i]
for c in compare_images:
    image = io.imread(input_path + c)
    image[:,:,:,0:2] *= 255
    io.imsave(output_path + c, image)
