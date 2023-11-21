# AHT thresholding
To load the AHT thresholder, download the AHT.py and the knee_locator.py files.

1) When creating an instance of the AHT thresholder (class name AutoHystThreshold) provide an input path which is either a single file path or a path to a folder of files.

2) To threshold the images use the run() function where the IHH bias and Window bias are selected by parameters that are each defaulted to 0 (neither bias applied).

3) The output binarised files will be saved as 'AHT_' + the image file name.

4) An output path can be provided to save the binarised image to a specific location.
