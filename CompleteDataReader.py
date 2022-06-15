import matplotlib.pyplot as plt
from skimage import io
import json
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from skimage.exposure import histogram
import math

json_path = "C:\\RESEARCH\\Mitophagy_data\\Complete CSV Data\\"
json_file_s = ["N1HighHystHistInfo.json", "HighHystHistInfo.json"]
image_path_base = "C:\\RESEARCH\\Mitophagy_data\\"

def image_path_builder():
    subset_position = ["N1", "N2", "N3", "N4"]
    image_path_list = [image_path_base + sp + "\\Preprocessed\\" for sp in subset_position]
    return image_path_list

image_paths = image_path_builder()

def deconvolved_image_path(sample_name):
    path = ""
    for p in image_paths:
        if isfile(join(p + sample_name)):
            path = p + sample_name
    return path

def json_extract():
    output_data = []
    for j in json_file_s:
        data = None
        f = open(json_path + j)
        data = json.load(f)
        output_data.append(data)
    return output_data

def extract_parameters(data_dict):
    for sample, timeframes in data_dict.items():
        frame_count = len(list(timeframes))
        sample_images = load_image(sample, frame_count)
        for t, details in timeframes.items():
            '''print(details["Intens"])
            print(details["Voxels"])'''
            frame_image = sample_images[t]
            fig, (ax1, ax2) = plt.subplots(2)
            slopes, slope_points = get_slope(str_to_float(details["Intens"]), str_to_float(details["Voxels"]))
            mving_avg = rolling_average(slopes, 8)
            test_centroid = centroid(str_to_float(details["Voxels"]), str_to_float(details["Intens"]))
            print("Timeframe:", t, "Centroid:", test_centroid, "Min Intensity:", int(details["Intens"][0]))
            fig.suptitle(sample + " timeframe " + str(t))
            ax1.plot(str_to_float(details["Intens"]), str_to_float(details["Voxels"]), color='b')
            ax2.plot(slope_points, mving_avg, color='b')
            ax1.axvline(x=float(details["Yen"]), color='r', label='Yen')
            ax1.axvline(x=float(details["Centroid"]), color='b', label='Centroid')
            ax1.axvline(x=test_centroid, color='k', label='Test Centroid')
            ax2.axvline(x=float(details["Yen"]), color='r', label='Yen')
            ax2.axvline(x=float(details["Centroid"]), color='b', label='Centroid')
            ax2.axvline(x=test_centroid, color='k', label='Test Centroid')
            plt.legend()
            instance_title = sample + " timeframe " + str(t)
            if 'Valid' in details:
                instance_title += " Valid: " + str(bool(int(details['Valid'])))
            render_image_sequence(frame_image, instance_title, int(details["Intens"][0]))
            plt.show()

def centroid(y, x=None, weights=None):
    if weights is None:
        max_y = max(y) - min(y)
        weighting = []
        for w in y:
            weighting.append((w-min(y))/max_y)
    else:
        weighting = weights

    if x is None:
        x_axis = np.linspace(0, len(y), len(y))
    else:
        x_axis = x
    dtf = []
    denom = []
    for n in range(len(weighting)):
        dtf.append(x_axis[n] * weighting[n])
        denom.append(weighting[n])
    centr = sum(dtf)/sum(denom)
    '''fig, ax = plt.subplots(1, 1)
    ax.plot(x_axis, y, color='b')
    ax.axvline(x=centr, color='k', label='Test Centroid')
    fig.suptitle("Test Centroid")'''
    return centr

def open_histogram(image, low_thresh):
    fig, (ax1, ax2) = plt.subplots(2)
    counts, centers = histogram(image)
    low_index = np.where(centers == int(low_thresh))[0][0]
    centers2 = centers[low_index:]
    counts2 = counts[low_index:]
    ax1.plot(centers, counts)
    ax2.plot(centers2, counts2)
    fig.suptitle("Raw Histogram")

def load_image(image_name, frame_count):
    """
    This function will open the preprocessed image. If the first dimension [time] does not match the frame_count provided (if not 1) then needs to be fixed.
    RGB will also be removed.
    :param image_name: The path to with the name of the sample file
    :param frame_count: The total number of time frames
    :return: The dictionary of images with the timeframe as the key
    """
    full_path = deconvolved_image_path(image_name)
    input_image = io.imread(full_path)
    # print("Original Image Shape", input_image.shape)
    if input_image.shape[-1] == 3 and input_image[-1] != input_image[-2] and len(input_image.shape) > 3:
        """ This will remove the rgb component """
        input_image = rgb_ave(input_image)
    image_set = {}
    slices = 1
    if frame_count == 1:
        image_set["0"] = input_image
        return image_set
    else:
        if input_image.shape[0] != frame_count:
            slices = int(input_image.shape[0]/frame_count)
        for t in range(frame_count):
            upper = slices+t
            z_stack = input_image[t:t+1]
            if z_stack.shape[0] == 1:
                z_stack = z_stack[0]
            # print("Z Stack Shape", z_stack.shape)
            image_set[str(t)] = z_stack
        return image_set

def rgb_ave(image):
    if len(image.shape) > 3 and image.shape[-1] == 3:
        return np.mean(image, axis=-1).astype('uint8')
    else:
        return image

def str_to_float(input_string_list):
    float_rep = [float(float_value) for float_value in input_string_list]
    return float_rep

def explore_results():
    final_data = json_extract()
    for fd in final_data:
        extract_parameters(fd)

def rolling_average(counts, window_size=10, rescale=False):
    adjusted = False
    if type(counts) is list and window_size > 1:
        new_list = []
        for n in range(0, int(window_size/2)):
            new_list.append(0)
        counts = new_list + counts + new_list
        adjusted = True
    df = pd.DataFrame(counts)
    moving_average = df.rolling(window_size, center=True).mean()
    average_results = flatten_list(iter(moving_average.values.tolist()))
    if adjusted:
        window_offset = int(window_size/2)
        average_results = average_results[window_offset:-window_offset]
        if rescale:
            # print("Prior to rescaling", average_results)
            for i in range(1, window_offset+1):
                average_results[i-1] = (average_results[i-1]*10)/i
                average_results[-i] = (average_results[-i]*10)/i
            # print("Rescaled results", average_results)
    return average_results

def get_slope(x, y):
    if len(x) != len(y):
        # print("Inconsistent x and y coordinates")
        return None, None
    else:
        slope_values = []
        for i in range(1, len(x), 1):
            slope = abs((y[i] - y[i-1])/(x[i] - x[i-1]))
            slope_values.append(slope)
        #new_x = np.linspace(0, len(slope_values), len(slope_values))
        new_x = x[1:]
        return slope_values, new_x

def flatten_list(nested_iter):
    new_list = []
    try:
        current_value = next(nested_iter)
        if type(current_value) is list:
            sub_iter = iter(current_value)
            new_list += flatten_list(sub_iter)
            resumed_result = flatten_list(nested_iter)
            if resumed_result is not None:
                new_list += resumed_result
        else:
            new_list += [current_value]
            next_value = flatten_list(nested_iter)
            if next_value is None:
                return new_list
            else:
                return new_list + next_value
    except StopIteration:
        return None
    return new_list

def render_image_sequence(image, image_type, low_thresh = 70):
    # print("Image Shape", image.shape)
    number_of_slices = image.shape[0]
    columns = math.ceil(number_of_slices/2)
    if columns == 0:
        rows = 1
        columns = number_of_slices
    elif columns <= 4:
        rows = 2
    elif 4 < columns < 6:
        rows = 3
        columns = math.ceil(number_of_slices/rows)
    else:
        rows = 4
        columns = math.ceil(number_of_slices / rows)

    fig, axarr = plt.subplots(rows, columns, figsize=(10, 5))
    fig.suptitle(image_type)
    fig.tight_layout()
    slice = 0

    if rows > 1:
        for row in range(rows):
            for column in range(columns):
                im_slice = image[slice, :, :]
                im_slice[np.where(im_slice < low_thresh)] = 0
                axarr[row, column].imshow(image[slice, :, :])
                axarr[row, column].set_title("Slice " + str(slice + 1))
                slice += 1
    else:
        for column in range(columns):
            axarr[column].imshow(image[slice, :, :])
            axarr[column].set_title("Slice " + str(slice + 1))
            slice += 1
    #plt.show()

if __name__ == "__main__":
    print("Hello")
    explore_results()
    '''test_array = np.array([0, 1, 2, 3, 4])
    for t in test_array:
        print(test_array[t:(t+1)])'''