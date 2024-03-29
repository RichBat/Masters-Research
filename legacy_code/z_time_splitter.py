import numpy as np
from skimage import data, io
from os import listdir
from os.path import isfile, join, exists
import tifffile
"""
This script exists to correct the axis of the preprocessed images. During the preprocessing process the z and t dimensions are incorrectly combined.
This will take the preprocessed image and compare it to the dimensions of the deconvolved image.

"""

decon_paths = ["C:\\RESEARCH\\Mitophagy_data\\N1\\Deconvolved\\", "C:\\RESEARCH\\Mitophagy_data\\N2\\Deconvolved\\",
               "C:\\RESEARCH\\Mitophagy_data\\N3\\Deconvolved\\", "C:\\RESEARCH\\Mitophagy_data\\N4\\Deconvolved\\"]
input_paths = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Input\\"
output_path = "C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"
input_files = [f for f in listdir(input_paths) if isfile(join(input_paths, f))]

decon_files = [[decon_path + f, f] for decon_path in decon_paths for f in listdir(decon_path) if isfile(join(decon_path, f))]

timeframes_to_keep = {"N3CCCP_4C=1.tif": [1, 10], "N3Con_1C=1.tif": [1, 8], "N3Con_1C=0.tif": [1, 2], "N4CCCP_1C=1.tif": [2],
                      "N4CCCP_2C=0.tif": [3], "N4CCCP_2C=1.tif": [10], "N2Rapa+CCCP+Baf_1C=1.tif": [6]}

def save_time_separated(segment, image, file):
    t_seg = image[segment]
    if len(t_seg.shape) > 3:
        t_seg = np.mean(t_seg, axis=-1)
    print(t_seg.shape)
    t_seg = t_seg.astype('uint8')
    tifffile.imwrite(output_path + file.split(".")[0] + "T=" + str(segment) + ".tif", t_seg, imagej=True, metadata={'axes': 'ZYX'})

for file in input_files:
    print("Sample", file)
    input_image = io.imread(input_paths + file)
    print(input_image.shape)
    decon_file = None
    for df in decon_files:
        if file == df[1]:
            decon_file = df[0]
            break
    if decon_file is not None:
        decon_image = io.imread(decon_file)
        image_list = []
        if len(decon_image.shape) != len(input_image.shape):
            t_dim = decon_image.shape[0]
            z_dim = decon_image.shape[1]
            for t in range(t_dim):
                t_lower = t*z_dim
                t_upper = t*z_dim + z_dim
                time_segment = input_image[t_lower:t_upper]
                if len(time_segment.shape) > 3 and time_segment.shape[-1] == 3:
                    time_segment = np.mean(time_segment, axis=-1)
                image_list.append(time_segment)
            input_image = np.stack(image_list, axis=0)
        for input_t in range(input_image.shape[0]):
            if file in timeframes_to_keep:
                if input_t in timeframes_to_keep[file]:
                    save_time_separated(input_t, input_image, file)
            else:
                if input_t == 0:
                    save_time_separated(input_t, input_image, file)

        print("Time Separated")


