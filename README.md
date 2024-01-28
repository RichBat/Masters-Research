# AHT thresholding
To load the AHT thresholder, download the AHT.py and the knee_locator.py files.

1) When creating an instance of the AHT thresholder (class name AutoHystThreshold) provide an input path which is either a single file path or a path to a folder of files.

2) To threshold the images use the run() function where the IHH bias and Window bias are selected by parameters that are each defaulted to 0 (neither bias applied).

3) The output binarised files will be saved as 'AHT_' + the image file name.

4) An output path can be provided to save the binarised image to a specific location.

# Adding Poisson and Gaussian noise
To add Poison and Gaussian noise run the script called 'adding_noise.py' wherein:

1) Provide the file source, the destination of the noise addition images, the Poisson lambdarange, and the Gaussian sigma range.

2) To generate images like those used in the metric analysis of this research, use binarized images in the file source.

# Evaluating metrics
WIP
