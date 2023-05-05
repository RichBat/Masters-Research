import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
from scipy import special, stats
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
from scipy import ndimage as ndi
import seaborn as sns
import tifffile
import warnings
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
from scipy import interpolate

from CleanThresholder import AutoThresholder

test_image_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"

def get_3d_image(image, depth_type=0, slice=None, object_colours=False):
    '''
    The provided image can be 2D or 3D but if 3D it needs to be specified what is done. This will produce a volumetric
    representation of either the z-stack (type=1) or a 2D slice with the pixel intensities providing depth (type=0).
    :param depth_type: Whether intensity is depth(0) or the actual z-stack is projected(1)
    :param image: The image array provided. Must be between 2 and 3 dims
    :param slice: If a 3D image array is provided and type 0 is selected then a slice must be designated
    :param object_colours: If false then a heatmap will be used for the object depths. If true then each independent
    structure will have a corresponding colour.
    :return:
    '''
    im_shape = image.shape
    if len(im_shape) > 3:
        raise Exception("There are more than 3 dimensions for this image")
    if depth_type != 0 and depth_type != 1:
        warnings.warn("A valid projection option was not selected. If 3D will default to type 1 and 2D to type 0")
        depth_type = 0 if len(im_shape) == 2 else 1

    if len(im_shape) == 3:
        if depth_type == 0 and slice is None:
            warnings.warn("3D image provided for 2D intensity projection without slice selected. Will default to centre slice")
            slice = math.ceil(image.shape[-1]/2)

    if depth_type == 0 and slice is not None:
        warnings.warn("A slice was provided for a 2D image projection. Slice was ignored.")

    if im_shape[0] or im_shape[1] > 1024:
        scaling_factor = int(max(im_shape[0], im_shape[1])/1024)
        image = scale_down(image, scaling_factor)

    if depth_type == 0:
        flat_image = image if len(im_shape) == 2 else image[slice]
        max_intensity = flat_image.max() + 1
        min_intensity = np.unique(flat_image)[0] if np.unique(flat_image)[0] != 0 else np.unique(flat_image)[1]
        #Check the intensity range of the flat image, the should be a non-zero minimum due to thresholding
        print(np.unique(flat_image))
        project_array = np.zeros(tuple(list(flat_image.shape) + [max_intensity-min_intensity]))
        nonzero_indices = np.nonzero(flat_image)
        flat_index = flat_image - min_intensity
        rows, cols = nonzero_indices[0], nonzero_indices[1]
        project_array[rows, cols, flat_index[rows, cols]] = flat_image[rows, cols]
        print(project_array.shape)
        plot_3d(project_array)
    '''For colour coding intensity projections we can get the object colours from 2D and then just assign to all 
    non-zero voxels sharing matching x and y coords'''

def plot_3d(image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    for i in range(image.shape[2]):
        X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        Z = np.full(X.shape, i)
        ax.contourf(X, Y, image[:, :, i], zdir='z', offset=i, cmap=plt.cm.viridis)

    # Add a colorbar to the plot
    fig.colorbar(ax.get_children()[0], ax=ax, orientation='horizontal')

    # Set the axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Microscope Z-stack in 3D")

    # Show the plot
    plt.show()

def scale_down(im, sf):
    out_height = im.shape[0] // sf  # Scale down by a factor of 2
    out_width = im.shape[1] // sf

    # Define the x and y coordinates for the output grid
    x_out = np.linspace(0, im.shape[1] - 1, out_width)
    y_out = np.linspace(0, im.shape[0] - 1, out_height)

    # Define the x and y coordinates for the input grid
    x_in = np.linspace(0, im.shape[1] - 1, im.shape[1])
    y_in = np.linspace(0, im.shape[0] - 1, im.shape[0])

    # Define the bicubic interpolation function
    interp_func = interpolate.interp2d(x_in, y_in, im, kind='cubic')

    # Evaluate the function at the output grid coordinates
    arr_out = interp_func(x_out, y_out).astype('uint8')

    return arr_out


if __name__ == "__main__":
    test_image = io.imread(test_image_path + "CCCP_1C=0T=0.tif")
    mask_thresh = 30
    test_mask = (test_image > mask_thresh).astype(int)
    masked_im = test_image * test_mask
    sliced_image = masked_im[4]
    get_3d_image(sliced_image)
