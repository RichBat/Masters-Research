import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
import tifffile
from knee_locator import KneeLocator
import time
from scipy import ndimage as ndi
from CleanThresholder import AutoThresholder

class threshRACC(AutoThresholder):

    def __init__(self, input_paths, power, steepness, option, deconv_paths=None):
        self.power = power
        self.steepness = steepness
        self.option = [option] if type(option) is not list else option
        self.sample_pairs = {}
        self.unmatched_samples = []
        self.invalid_samples = []
        AutoThresholder.__init__(self, input_paths, deconv_paths)

    def apply_RACC(self, save_path=None, threshold_list=None):
        pairs = self._get_sample_pairs()
        print(pairs)
        for k, p in pairs.items():
            print("Sample:", k)
            if type(p) is tuple:
                image1, thresholds1 = self._threshold_one(p[0][0], p[0][1])
                list_set1 = set(list(thresholds1))
                image2, thresholds2 = self._threshold_one(p[1][0], p[1][1])
                list_set2 = set(list(thresholds2))
                print(list_set2)
                shared_values = list_set1.intersection(list_set2)
                print(shared_values)
                for i in shared_values:
                    racc_results = self.RACC([thresholds1[i], thresholds2[i]], image1[i], image2[i], 45, 95, False)
                    io.imshow(np.amax(racc_results, axis=0))
                    plt.show()
                    if save_path is not None:
                        io.imsave(save_path + k + ".tif")

    def _threshold_one(self, file_path, filename):
        image = io.imread(file_path)
        print("Orig Shape", image.shape)
        gray_image = self._grayscale(image)
        image_set = self._timeframe_sep(gray_image, filename)
        image_holder = np.zeros_like(image_set)
        image_thresholds = {}
        print("Timeframes:", image_set.shape[0])
        for i in range(image_set.shape[0]):
            high_threshes = self._specific_thresholds(image_set[i], ["Inverted"], self.option, steepness=self.steepness, power=self.power)
            print(high_threshes)
            if high_threshes is not None:
                for thresh_type, thresholds in high_threshes.items():
                    thresholds_key = list(thresholds)[0]
                    image_mask = self._threshold_image(image_set[i], thresholds[thresholds_key][0], thresholds[thresholds_key][1]).astype("int")
                    thresholed_image = (image_set[i] * image_mask).astype("uint8")
                    image_holder[i] = thresholed_image
                    image_thresholds[i] = thresholds[thresholds_key][0]
        return image_holder, image_thresholds

    def RACC(self, thresholds, ch1, ch2, value, percentage, calculated):
        # Preprocessing and setup
        # threshCh1 = int(self.ch1ThreshLE.text())
        # threshCh2 = int(self.ch2ThreshLE.text())
        # All of the self.   .text() objects are inherited from the GUI classes. They need to be replaced currently with arguement parameters for the function
        # or from a new function to extract those parameters from a text file
        threshCh1 = thresholds[0]
        threshCh2 = thresholds[1]
        penFactor = value
        percToInclude = percentage / 100

        print("Process with values:\nThres Ch1: {}\nThres Ch2: {}".format(threshCh1, threshCh2))
        print("Penalization factor (theta): {}\nPercentage to include: {}".format(penFactor, percToInclude * 100))

        ch1Stack = ch1
        ch2Stack = ch2

        maxIntensity1 = np.max(ch1Stack)
        maxIntensity2 = np.max(ch2Stack)
        maxInt = np.max((maxIntensity1, maxIntensity2))

        print("\nch1Stack.shape", ch1Stack.shape)
        print("ch2Stack.shape", ch2Stack.shape)

        timeframes = 1

        if (maxInt > 255):
            print("Max intensity greater than 255 ({}), adjusted".format(maxInt))
            ch1Stack = ch1Stack / maxInt * 255
            ch2Stack = ch2Stack / maxInt * 255

        # originalShape = ch1Stack.shape

        ''' isStack = True  # 3D stack if True, otherwise single slice 2D image
        if (ch1Stack.shape != ch2Stack.shape):
            print("\nERROR: stack shapes are not the same, cannot continue.")
            self.showErrorDialog(
                "Stack shapes are not the same, cannot continue.\n{} vs {}".format(ch1Stack.shape, ch2Stack.shape))
            return'''

        print("Image Stack dimensions: ", len(ch1Stack.shape))
        #################################
        # Calculate RACC

        print("\n\nSTART of RACC calculation:")

        theta = penFactor * math.pi / 180.0
        xMax = -1
        yMax = -1

        Imax = 255
        texSize = 256

        #####################
        # CALCULATE averages and covariances

        valuesAboveThreshCh1 = ch1Stack[ch1Stack >= threshCh1]
        valuesAboveThreshCh2 = ch2Stack[ch2Stack >= threshCh2]

        averageCh1 = np.average(valuesAboveThreshCh1)
        averageCh2 = np.average(valuesAboveThreshCh2)

        print("\nAverage Ch1: {}\nAverage Ch2: {}".format(averageCh1, averageCh2))
        ch1Stack_shape = ch1Stack.shape
        filteredCh1Stack = ch1Stack
        filteredCh2Stack = ch2Stack
        filteredCh1Stack[filteredCh1Stack < threshCh1] = 0
        filteredCh2Stack[filteredCh2Stack < threshCh2] = 0

        filteredCh1Stack = filteredCh1Stack.ravel()
        filteredCh2Stack = filteredCh2Stack.ravel()

        covariance = np.cov(filteredCh1Stack, filteredCh2Stack)
        varXX = covariance[0, 0]
        varYY = covariance[1, 1]
        varXY = covariance[0, 1]

        print("\nCovariance(xx): {}\nCovariance(yy): {}\nCovariance(xy): {}".format(varXX, varYY, varXY))

        #####################
        # CALCULATE B0 and B1

        lamb = 1  # special case of Deming regression
        val = lamb * varYY - varXX

        B0 = 0
        B1 = 0

        if (varXY < 0):
            print("\nThe covariance is negative")
            B1 = (val - math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)
        else:
            B1 = (val + math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)

        B0 = averageCh2 - B1 * averageCh1

        print("\nB0 = {}  B1 = {}".format(B0, B1))

        #####################
        # CALCULATE p0 and p1

        p0 = np.zeros(2)
        p1 = np.zeros(2)
        p0[0] = 0
        p0[1] = 0
        p1[0] = 255
        p1[1] = 255
        if(calculated):
            # For P0
            print("P values calculated")
            if (threshCh2 <= threshCh1 * B1 + B0):
                p0[0] = threshCh1
                p0[1] = threshCh1 * B1 + B0
            elif (threshCh2 > threshCh1 * B1 + B0):
                p0[0] = (threshCh2 - B0) / B1
                p0[1] = threshCh2

            # For P1
            if (B0 >= Imax * (1 - B1)):
                p1[0] = (Imax - B0) / B1
                p1[1] = Imax
            elif (B0 < Imax * (1 - B1)):
                p1[0] = Imax
                p1[1] = Imax * B1 + B0

        print("\nP0 = {}  P1 = {}".format(p0, p1))

        #####################
        # CALCULATE xMax

        totalVoxelCount = 0
        colorMapFrequencyX = np.zeros(texSize)
        colorMapFrequencyY = np.zeros(texSize)

        overlappingSection = np.multiply(np.clip(filteredCh1Stack, 0, 1), np.clip(filteredCh2Stack, 0, 1)) * Imax
        reducedCh1Stack = filteredCh1Stack[overlappingSection > 0]
        reducedCh2Stack = filteredCh2Stack[overlappingSection > 0]
        print("\nOverlapping section: ", overlappingSection)
        print(
            "\nFull size was {} reduced colocalized size is {}. Remaining percentage {}%".format(filteredCh1Stack.shape,
                                                                                                 reducedCh1Stack.shape,
                                                                                                 reducedCh1Stack.shape[
                                                                                                     0] /
                                                                                                 filteredCh1Stack.shape[
                                                                                                     0] * 100))

        totalVoxelCount = reducedCh2Stack.shape[0]
        print("Total number of voxels: ", totalVoxelCount)
        qMat = np.stack((reducedCh1Stack, reducedCh2Stack))

        kMat = ((p1[1] - p0[1]) * (qMat[0] - p0[0]) - (p1[0] - p0[0]) * (qMat[1] - p0[1])) / (
                (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0]))

        xMat = qMat[0] - kMat * (p1[1] - p0[1])
        yMat = qMat[1] + kMat * (p1[0] - p0[0])

        fracXMat = (xMat - p0[0]) / (p1[0] - p0[0])
        fracXMat = (np.clip(fracXMat, 0, 1) * Imax).astype(int)
        unique, counts = np.unique(fracXMat, return_counts=True)

        if (len(unique) >= 256):
            print("Note (this should never happen): Had to crop values to 255, max was ", len(unique))
            unique = unique[0:255]
            counts = counts[0:255]
        colorMapFrequencyX[unique] = counts

        fracYMat = (yMat - p0[1]) / (p1[1] - p0[1])
        fracYMat = (np.clip(fracYMat, 0, 1) * Imax).astype(int)
        unique, counts = np.unique(fracYMat, return_counts=True)
        if (len(unique) >= 256):
            print("Note (this should never happen): Had to crop values to 255, max was ", len(unique))
            unique = unique[0:255]
            counts = counts[0:255]
        colorMapFrequencyY[unique] = counts

        cumulativeTotalX = 0
        cumulativeTotalY = 0

        for i in range(0, texSize):
            cumulativeTotalX += colorMapFrequencyX[i]
            cumulativeTotalY += colorMapFrequencyY[i]

            if (cumulativeTotalX / totalVoxelCount >= percToInclude):
                xMax = (p1[0] - p0[0]) * (i / texSize) + p0[0]
                break
            elif (cumulativeTotalY / totalVoxelCount >= percToInclude):
                yMax = (p1[1] - p0[1]) * (i / texSize) + p0[1]
                break

        # some verification code
        if (xMax < 0 and yMax < 0):
            if (B1 < 1):
                xMax = Imax
            else:
                yMax = Imax

        print("\nMax X: {} / {} ({}%)".format(cumulativeTotalX, totalVoxelCount,
                                              cumulativeTotalX / totalVoxelCount * 100));
        print(
            "Max Y: {} / {} ({}%)".format(cumulativeTotalY, totalVoxelCount, cumulativeTotalY / totalVoxelCount * 100));
        print("X_Max: {}  Y_Max: {}".format(xMax, yMax));

        #####################
        # CALCULATE distance threshold (variant of binary search)

        distanceCount = 0
        dThresh = 0
        tryCount = 0
        dMin = 0
        dMax = Imax

        print("\nCalc Distance iteration: ", end="")
        print(tryCount, Imax)
        while (tryCount <= Imax):
            tryCount += 1
            print("{}  ".format(tryCount), end="")

            dThresh = dMin + (dMax - dMin) / 2.0

            dMat = ((abs(
                (p1[1] - p0[1]) * qMat[0] - (p1[0] - p0[0]) * qMat[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt(
                (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])))
            distanceCount = np.where(dMat < dThresh)[0].shape[0]


            if (distanceCount / totalVoxelCount == percToInclude):
                print("Free1")
                break
            elif (distanceCount / totalVoxelCount < percToInclude):
                dMin = int(dThresh) + 1
            else:
                dMax = int(dThresh) - 1

            if (dMin == dMax):
                print("Free2")
                break

            print("\n\nDistance threshold for {}% = {} (within {} times)".format(percToInclude * 100, dThresh, tryCount));
            if (calculated):
                if (xMax != -1):
                    p1[0] = xMax
                    p1[1] = B1 * p1[0] + B0
                else:
                    p1[1] = yMax
                    p1[0] = (p1[1] - B0) / B1

            print("p_max for {}% = {}".format(percToInclude * 100, p1))

            #####################
            # Generate Ci greyscale map
            p0 = p0 / Imax
            p1 = p1 / Imax

            filteredCh1Stack[overlappingSection == 0] = 0
            filteredCh2Stack[overlappingSection == 0] = 0

            output = np.zeros_like(filteredCh1Stack)
            qMat = np.stack((filteredCh1Stack, filteredCh2Stack)) / Imax
            kMat = ((p1[1] - p0[1]) * (qMat[0] - p0[0]) - (p1[0] - p0[0]) * (qMat[1] - p0[1])) / (
                    (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0]))
            xMat = qMat[0] - kMat * (p1[1] - p0[1])
            dMat = ((abs(
                (p1[1] - p0[1]) * qMat[0] - (p1[0] - p0[0]) * qMat[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt(
                (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])))

            condition = np.logical_and((dMat * (p1[0] - p0[0]) * math.tan(theta) + p0[0] < xMat), (xMat < p1[0]))
            output[condition] = ((xMat[condition] - p0[0]) / (p1[0] - p0[0]) - dMat[condition] * math.tan(theta)) * Imax
            condition = xMat >= p1[0]
            output[condition] = np.clip(((1 - dMat[condition] * math.tan(theta)) * Imax), 0, 255)
            condition = np.logical_or((xMat <= dMat * (p1[0] - p0[0]) * math.tan(theta) + p0[0]), (dMat > dThresh / Imax))
            output[condition] = 0

            # This is for the colour mapping while the GUI is removed:
            '''colormap = io.imread("magmaLine.png").astype(np.uint8)[0, :, 0:3]
            colormap[0] = np.zeros(3)
            grayColormap = output.reshape(ch1Stack.shape)
            output = colormap[output]
            output = output.reshape(originalShape)'''

        output = output.reshape(ch1Stack.shape)
        grayColormap = output

        return grayColormap

    def _get_sample_pairs(self):
        im_path_keys = list(self.image_paths)
        sample_set = {}
        sample_combos = {}
        for f in im_path_keys:
            full_sample_name = f.split("C=")
            sample_name = full_sample_name[0]
            channel_name = full_sample_name[1].split(".")[0]
            if "T" in channel_name:
                channel_name = channel_name.split("T")[0]
            if sample_name not in sample_set:
                sample_set[sample_name] = {}
            sample_set[sample_name][channel_name] = (self.image_paths[f], f)
        for k, v in sample_set.items():
            if '1' not in v:
                self.unmatched_samples.append(k)
            elif len(v) == 2:
                sample_combos[k] = (v['1'], v['0'])
            elif len(v) == 3:
                sample_combos[k] = [(v['2'], v['1']), (v['2'], v['0'])]
            else:
                self.unmatched_samples.append(k)
        return sample_combos



if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\RACCtest\\Input\\"]
    output_path = "C:\\RESEARCH\\Mitophagy_data\\RACCtest\\Output\\"
    deconv_path = ["C:\\RESEARCH\\Mitophagy_data\\N3\\Deconvolved\\"]
    pipeline = threshRACC(input_path, 1, 6, 0)
    pipeline.apply_RACC(output_path)