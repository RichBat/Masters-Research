import numpy as np
import matplotlib.pyplot as plt
import math
from mayavi import mlab
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
from skimage import morphology

from CleanThresholder import AutoThresholder

test_image_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"

def get_3d_image(image, depth_type=0, slice=None, object_colours=False, image_mask=None):
    '''
    The provided image can be 2D or 3D but if 3D it needs to be specified what is done. This will produce a volumetric
    representation of either the z-stack (type=1) or a 2D slice with the pixel intensities providing depth (type=0).
    :param depth_type: Whether intensity is depth(0) or the actual z-stack is projected(1)
    :param image: The image array provided. Must be between 2 and 3 dims
    :param slice: If a 3D image array is provided and type 0 is selected then a slice must be designated
    :param object_colours: If false then a heatmap will be used for the object depths. If true then each independent
    structure will have a corresponding colour.
    :param depth_type: The threshold mask used to remove low intensity pixels from the image. This will be used to
    make up for non-zero background after descaling with interpolation.
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

    '''if im_shape[0] or im_shape[1] > 2048:
        scaling_factor = int(max(im_shape[0], im_shape[1])/1024)
        #scaled_image = scale_down(image, scaling_factor)
        scaled_image = scale_down2(image, scaling_factor)
        if image_mask is not None:
            image_mask = scale_down(image_mask, scaling_factor)
        image = scaled_image'''

    if depth_type == 0:
        flat_image = image if len(im_shape) == 2 else image[slice]
        max_intensity = flat_image.max() + 1
        min_intensity = np.unique(flat_image)[0] if np.unique(flat_image)[0] != 0 else np.unique(flat_image)[1]
        #Check the intensity range of the flat image, the should be a non-zero minimum due to thresholding
        '''print(np.unique(flat_image))
        io.imshow(flat_image)
        plt.show()'''
        '''project_array = np.zeros(tuple(list(flat_image.shape) + [max_intensity-30]))
        # perhaps make all empty regions -1 instead?
        nonzero_indices = np.nonzero(flat_image)
        flat_index = flat_image - 30
        # rows, cols = nonzero_indices[0], nonzero_indices[1]
        indices = np.argwhere(flat_image > 30)
        print(indices.shape)
        mask = np.zeros_like(project_array, dtype=bool)

        for ind in indices:
            mask[ind[0], ind[1], :flat_image[ind[0], ind[1]]] = True
        project_array[mask] = 1
        #project_array[rows, cols, flat_index[rows, cols]] = flat_image[rows, cols]
        #Take the flat array and extend the value in ranges
        height_range = np.arange(30, max_intensity)
        project_array = project_array * height_range'''
        '''flattened = np.amax(project_array, axis=-1)
        io.imshow(flattened)
        plt.show()'''
        # plot_3D_Maya2(project_array)
        plot_as_surface(flat_image, 9)
        # test_surf()
    '''For colour coding intensity projections we can get the object colours from 2D and then just assign to all 
    non-zero voxels sharing matching x and y coords'''

def plot_3d(image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    for i in range(image.shape[2]):
        X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        Z = np.full(X.shape, i)
        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_under(alpha=0)
        ax.contourf(X, Y, image[:, :, i], zdir='z', offset=i, cmap=cmap, vmin=30, vmax=image.max())
    # Add a colorbar to the plot
    fig.colorbar(ax.get_children()[0], ax=ax, orientation='horizontal')

    # Set the axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Microscope Z-stack in 3D")

    # Show the plot
    plt.show()


def plot_3D_Maya(image):
    cont3d = mlab.contour3d(image, opacity=1, vmin=30, colormap='viridis')
    mlab.colorbar(cont3d)
    mlab.outline(cont3d)
    mlab.show()


def plot_3D_Maya2(image):
    vol3d = mlab.pipeline.volume(mlab.pipeline.scalar_field(image), vmin=30)
    mlab.colorbar(vol3d)
    mlab.outline(vol3d)
    mlab.show()


def plot_as_surface(image, minimum=None):
    image = image.astype(float)
    mask = np.ma.less_equal(image, minimum) if minimum is not None else None
    mask[np.isnan(mask)] = True
    surf_vol = mlab.surf(image, vmin=minimum, warp_scale="auto", colormap='viridis', transparent=True, mask=mask)
    flat_test = np.ones_like(image)*100
    surf_vol_2 = mlab.surf(flat_test, opacity=0.2)
    mlab.colorbar(surf_vol)
    mlab.outline(surf_vol)
    mlab.axes(surf_vol)
    mlab.show()

def hysteresis_surface(image, low_thr, high_thr):
    thresholded = apply_hysteresis_threshold(image, low_thr, high_thr)
    thresholded = np.logical_not(thresholded)
    image = image.astype(float)
    surf_vol = mlab.surf(image, warp_scale="auto", colormap='viridis', transparent=True, mask=thresholded)
    mlab.colorbar(surf_vol)
    mlab.outline(surf_vol)
    mlab.axes(surf_vol)
    mlab.show()

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = mlab.surf(x, y, f)
    mlab.show()

def labelled_mesh(image, labels):
    '''
    This will plot a 3D mesh (similar to a surface) but will apply a custom color to the surface based on the labels
    :param image: A 2D array representing the image intensities
    :param labels: A 2D array representing the image labels
    :return:
    '''

    X, Y = np.meshgrid(range(image.shape[0]), range(image.shape[1]), indexing='ij')
    # The image is the intensity Z
    mlab.mesh(X, Y, image, scalars=labels)
    mlab.show()

def hyst_3d(image, low_thr, high_thr, show_thresh=0, label_thresh=None, thresholded_regions=None, show_orig=True):
    '''
    This function will create a 3D mesh image. The low and high thresholds must be supplied with a coloured flat plane
    optionally visualised for the thresholds; the labelling of the regions based on whether they are above, below or
    between the thresholds using RGB gradients (Red for above, Green for between, Blue for below); the option to output
    the 3D image of only the region around the thresholds; and the option to toggle whether the original image with
    threshold surfaces are shown at all.
    :param image:
    :param low_thr:
    :param high_thr:
    :param show_thresh:
    :param label_thresh:
    :param thresholded_regions:
    :param show_orig:
    :return:
    '''
    image = image.astype(float)

    # thresh_image = (apply_hysteresis_threshold(image, low_thr, high_thr).astype(float) * image).astype(float)
    # foot = morphology.disk(6)
    # high_only = morphology.binary_dilation((image >= high_thr), foot)
    # low_only = morphology.binary_dilation((image >= low_thr), foot)
    # mask = np.ma.less_equal(image, low_thr)
    # mask[np.isnan(mask)] = True

    X, Y = np.meshgrid(range(image.shape[0]), range(image.shape[1]), indexing='ij')

    if show_orig:

        mlab.mesh(X, Y, image, colormap='gist_earth')
        if show_thresh == 0 or show_thresh == 1:
            low_surf = (np.ones_like(image) * low_thr).astype(float)
            low_inter = mlab.mesh(X, Y, low_surf, scalars=low_surf, colormap='viridis', opacity=0.4, line_width=0.1)
            low_inter.module_manager.scalar_lut_manager.use_default_range = False
            low_inter.module_manager.scalar_lut_manager.data_range = image.min(), image.max()
        if show_thresh == 0 or show_thresh == 2:
            high_surf = (np.ones_like(image) * high_thr).astype(float)
            high_inter = mlab.mesh(X, Y, high_surf,  scalars=high_surf, colormap='viridis', opacity=0.4, line_width=0.1)
            high_inter.module_manager.scalar_lut_manager.use_default_range = False
            high_inter.module_manager.scalar_lut_manager.data_range = image.min(), image.max()
        mlab.show()

    if label_thresh is not None:

        def get_color_range(bottom, top):
            min_pixels, max_pixels = image > bottom, image < top
            overlap_range = np.logical_and(min_pixels, max_pixels)
            minimum, maximum = image[overlap_range].min(), image[overlap_range].max()
            span = int(255 / (top - bottom))
            colour_mapping = np.arange(0, 255, span)
            print(colour_mapping, colour_mapping.shape, minimum, maximum, span)
            return colour_mapping, bottom

        rgb_array = np.stack([np.zeros_like(image), np.zeros_like(image), np.zeros_like(image)], axis=-1)

        if label_thresh == 0:
            image = image.astype(int)
            current_colour = image > high_thr # This is to use for index the above high thresh regions
            colour_range, intensity_offset = get_color_range(high_thr+1, 256)
            rgb_array[current_colour, 0] = colour_range[image[current_colour]-intensity_offset]


            current_colour = np.logical_and(image > low_thr, np.logical_not(current_colour)) # This will be for in-between
            colour_range, intensity_offset = get_color_range(low_thr + 1, high_thr)
            rgb_array[current_colour, 1] = colour_range[image[current_colour] - intensity_offset]

            current_colour = np.less_equal(image, low_thr)
            colour_range, intensity_offset = get_color_range(0, low_thr)
            print(image[current_colour].max())
            print(colour_range, intensity_offset)
            rgb_array[current_colour, 2] = colour_range[image[current_colour] - intensity_offset]

        mlab.mesh(X, Y, image, scalars=rgb_array)
        #mlab.mesh(X, Y, image)
        mlab.show()


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
    interp_func = interpolate.interp2d(x_in, y_in, im, kind='linear')

    # Evaluate the function at the output grid coordinates
    arr_out = interp_func(x_out, y_out).astype('uint8')

    return arr_out


def scale_down2(im, sf):

    # Compute new dimensions
    width, height = im.shape
    scale_factor = sf
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    new_image = cv2.resize(im.astype('uint8'), (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return new_image

def get_structure_labels(image, lw_thrsh):
    mask_low = image > lw_thrsh
    labels_low, num_labels = ndi.label(mask_low)
    valid_structures = np.stack([labels_low, image * (mask_low.astype('int'))],
                                axis=-1)  # The two labels have been stacked
    valid_structures = np.reshape(valid_structures, (
    -1, valid_structures.shape[-1]))  # The dual label image has been flattened save for the label pairs
    print(valid_structures.shape)
    sort_indices = np.argsort(valid_structures[:, 0])
    print("Index sorting", sort_indices)
    valid_structures = valid_structures[sort_indices]
    label_set, start_index, label_count = np.unique(valid_structures[:, 0], return_index=True, return_counts=True)
    end_index = start_index + label_count
    max_labels = np.zeros(tuple([len(label_set), 2]))
    canvas_image = np.zeros_like(labels_low)
    for t in range(len(label_set)):
        max_labels[t, 0] = label_set[t]
        max_labels[t, 1] = valid_structures[slice(start_index[t], end_index[t]), 1].max()
        # canvas_image += (labels_low == label_set[t]).astype('int') * valid_structures[slice(start_index[t], end_index[t]), 1].max()
    value_mapping = max_labels[:, 1]
    canvas_image = value_mapping[labels_low]

if __name__ == "__main__":
    test_image = io.imread(test_image_path + "CCCP_1C=1T=0.tif")
    mask_thresh = 9
    test_mask = (test_image > mask_thresh).astype(int)
    masked_im = test_image
    sliced_image = masked_im[4]
    sliced_image = sliced_image[1131:2829, 536:2800]
    labels = np.ones_like(sliced_image)
    # get_structure_labels(sliced_image, 20)
    # plot_as_surface(sliced_image, 9)
    hyst_3d(sliced_image, 20, 140, show_orig=True, show_thresh=0)
    # labelled_mesh(sliced_image, labels)
    #hysteresis_surface(sliced_image, 9, 144)
    #get_3d_image(sliced_image, image_mask=test_mask[4])
