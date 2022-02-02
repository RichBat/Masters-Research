from os import listdir
from os.path import isfile, join, exists

import pandas as pd

from skimage import data, io
from skimage.filters import apply_hysteresis_threshold
from skimage.exposure import histogram
import matplotlib.pyplot as plt

import numpy as np
import math

def main():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    images = [input_path + f for f in listdir(input_path) if isfile(join(input_path, f))]
    samples = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    print(images)
    set_of_samples = {}
    channels_per_sample = {}
    print("Samples:", samples)
    for s in samples:
        key = s.split("=")[0][:-1]
        set_of_samples[key] = []
        if key not in channels_per_sample.keys():
            channels_per_sample[key] = 1
        else:
            channels_per_sample[key] += 1
    print(channels_per_sample)
    #print(set_of_samples)
    for i in images:
        filename = i.split(sep=input_path)[1].split("=")[0][:-1]
        print(filename)
        mask, threshold = thresholding(i, 10, 4)
        set_of_samples[filename].append([masking(io.imread(i), mask), threshold])
        print(len(set_of_samples[filename]))
        if len(set_of_samples[filename]) == channels_per_sample[filename]:
            #Need to correctly pair up the masks and the intensity images. Currently only the masks are stored!!!
            thresholded = set_of_samples.pop(filename, None)
            #Currently threshold arrays are all masked thus they are segmented intensity images. Binary images makes no sense as there is then literally zero correlation information
            coloc_results = RACC([thresholded[0][1], thresholded[1][1]], thresholded[0][0], thresholded[1][0], 45, 95, False)
            print("Coloc Time")
            print(coloc_results.shape)
            io.imsave(output_path + filename + ".tif", coloc_results)
    print("Set of samples:", set_of_samples)
    print(samples)
    return

def convertRGB(channel, Array):
    newShape = list(Array.shape)
    newShape.append(3)
    #newArray = np.zeros(shape=newShape)
    stackTest1 = np.zeros(shape=Array.shape)
    stackTest2 = np.zeros(shape=Array.shape)
    if channel == 0:
        stackedArray = np.stack((Array, stackTest1, stackTest2), axis=-1)
    elif channel == 1:
        stackedArray = np.stack((stackTest1, Array, stackTest2), axis=-1)
    else:
        print("invalid channel")
        return
    '''for z in range(Array.shape[0]):
        for w in range(Array.shape[1]):
            for h in range(Array.shape[2]):
                newArray[z][w][h][channel] = Array[z][w][h]'''
    return stackedArray





def thresholding(i, movingAverageFrame, cutOffSlope):
    img = io.imread(i)
    if len(img.shape) <= 3:
        low, high = determineHysteresisThresholds(img=img, movingAverageFrame=movingAverageFrame, cutOffSlope=cutOffSlope)
        normalize_img = img / np.max(img)
        thresholded = hysteresisThresholdingStack(normalize_img, low, high)
        return thresholded, low
    else:
        #print("TIME", img.shape)
        threshold_timepoints = []
        #print("Number of images: ", img.shape[0])
        lows = []
        for t in range(img.shape[0]):
            #print("Index: ", t)
            low, high = determineHysteresisThresholds(img=img[t], movingAverageFrame=movingAverageFrame, cutOffSlope=cutOffSlope)
            normalize_img = img[t] / np.max(img[t])
            threshold_timepoints.append(hysteresisThresholdingStack(normalize_img, low, high))
            lows.append(low)
        thresholded = np.stack(threshold_timepoints)
        print(type(thresholded))
        return thresholded, lows
        # NB!!!!!!! Need to Try catch and fix dimensions for FiJi (time + z combined!!)

def hysteresisThresholdingStack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def determineHysteresisThresholds(img, bins=256, movingAverageFrame=20, cutOffSlope=2, highVal=0.95): #This function is from Rensu's MEL (Make sure to reference)
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]

    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()

    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0

    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        if movingAverage[0][i-10]/movingAverage[0][i+10] >= cutOffSlope:
              useIntensityLow = i
              #print("Low intensity to be used: ", useIntensityLow)
              #print("High intensity to be used: ", (1.0-(1.0-useIntensityLow/bins)/2)*bins)
              break

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))

def RACC(thresholds, ch1, ch2, value, percentage, calculated):
    # Preprocessing and setup
    # threshCh1 = int(self.ch1ThreshLE.text())
    # threshCh2 = int(self.ch2ThreshLE.text())
    # All of the self.   .text() objects are inherited from the GUI classes. They need to be replaced currently with arguement parameters for the function
    # or from a new function to extract those parameters from a text file
    threshCh1 = thresholds[0]
    threshCh2 = thresholds[1]
    penFactor = value
    percToInclude = percentage / 100
    # penFactor = int(self.thetaLE.text())
    # percToInclude = int(self.percentageLE.text()) / 100.0

    print("Process with values:\nThres Ch1: {}\nThres Ch2: {}".format(threshCh1, threshCh2))
    print("Penalization factor (theta): {}\nPercentage to include: {}".format(penFactor, percToInclude * 100))
    # self.channel1_FileLE is an inherited object from PyQt5. This is the object used for the widgets in the gui. The file name will be the path to the image
    # contains values 0-255
    # ch1Stack = io.imread(self.channel1_FileLE.text())
    # ch2Stack = io.imread(self.channel2_FileLE.text())
    ch1Stack = convertRGB(channel=0, Array=ch1)
    ch2Stack = convertRGB(channel=1, Array=ch2)

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
    ch1stack = ch1Stack
    ch2stack = ch2Stack
    outputList = []
    if (len(ch1Stack.shape) == 4):  # assuming (slices,x,y,RGB/RGBA)
        # remove alpha channel if exists
        ch1Stack = ch1Stack[:, :, :, 0:3]
        ch2Stack = ch2Stack[:, :, :, 0:3]

        originalShape = ch1Stack.shape

        ch1Stack = np.amax(ch1Stack, axis=3)
        ch2Stack = np.amax(ch2Stack, axis=3)

        # visualization
        """if (self.visualizeInputFilesCheckBox.isChecked()):
            self.mayavi_input3D.show()

            self.mayavi_input3D.visualization = VisualizationInput()
            self.mayavi_input3D.visualization.fullStack3D = ch1Stack / 4 + (np.ceil(ch2Stack / 4)) * 64

            self.mayavi_input3D.refresh()"""

        isStack = True

        print("\nExtracted only intensity values")
        print("ch1Stack.shape", ch1Stack.shape)
        print("ch2Stack.shape", ch2Stack.shape)


    elif (len(ch1Stack.shape) == 3):  # assuming (x, y, RGB/RGBA)
        # remove alpha channel if exists
        ch1Stack = ch1Stack[:, :, 0:3]
        ch2Stack = ch2Stack[:, :, 0:3]

        '''# visualization
        if (self.visualizeInputFilesCheckBox.isChecked()):
            # self.mainLayout.setColumnMinimumWidth(1,490)
            self.static_canvasInput2D.show()
            self._static_ax.clear()
            self._static_ax.imshow(ch1Stack + ch2Stack)
            self._static_ax.axis('off')
            self._static_ax.figure.canvas.draw()'''

        originalShape = ch1Stack.shape

        ch1Stack = np.amax(ch1Stack, axis=2)
        ch2Stack = np.amax(ch2Stack, axis=2)

        isStack = False

        print("\nExtracted only intensity values")
        print("ch1Stack.shape", ch1Stack.shape)
        print("ch2Stack.shape", ch2Stack.shape)

    #################################
    # Calculate RACC

    print("\n\nSTART of RACC calculation:")

    theta = penFactor * math.pi / 180.0
    dThresh = 255
    xMax = -1
    yMax = -1

    Imax = 255
    texSize = 256

    #####################
    # CALCULATE averages and covariances

    valuesAboveThreshCh1 = ch1Stack[np.where(ch1Stack >= threshCh1)]
    valuesAboveThreshCh2 = ch2Stack[np.where(ch2Stack >= threshCh2)]

    averageCh1 = np.average(valuesAboveThreshCh1)
    averageCh2 = np.average(valuesAboveThreshCh2)

    print("\nAverage Ch1: {}\nAverage Ch2: {}".format(averageCh1, averageCh2))

    filteredCh1Stack = np.copy(ch1Stack)
    filteredCh2Stack = np.copy(ch2Stack)
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

        # iterative approach (MUCH slower)
        #            distanceCount = 0
        #            for i in range(0,reducedCh1Stack.shape[0]):
        #                val_1 = reducedCh1Stack[i]
        #                val_2 = reducedCh2Stack[i]
        #
        #                #if val_1 > 0 and val_2 > 0: #should always be true
        #                qi = np.array([val_1, val_2])
        #
        #                #https://stackoverflow.com/questions/1811549/perpendicular-on-a-line-from-a-given-point/1811636#1811636
        #                k = ((p1[1] - p0[1]) * (qi[0] - p0[0]) - (p1[0] - p0[0]) * (qi[1] - p0[1])) / ((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) *(p1[0] - p0[0]))
        #                xi = qi[0] - k * (p1[1] - p0[1])
        #
        #                d = ((abs((p1[1] - p0[1]) * qi[0] - (p1[0] - p0[0]) * qi[1] + p1[0] * p0[1] - p1[1] * p0[0])) / math.sqrt((p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[0] - p0[0]) * (p1[0] - p0[0])));
        #
        #                if (d < dThresh):
        #                    distanceCount += 1

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
    print("\nFINISHED processing")
    '''
    if (self.grayscaleOutput.isChecked()):
        print("\nOutput in Grayscale")
        output = output.reshape(ch1Stack.shape)
        grayColormap = output
    else:
        print("\nOutput in Color")
        grayColormap = output.reshape(ch1Stack.shape)
        output = self.colormap[output]
        output = output.reshape(originalShape)

    print("Output shape: ", output.shape)

    #The output section needs to be evaluated based on the structuring of the system output
    '''

def masking(data, mask):
    #Both are expected to be numpy arrays of the same shape. Timepoint versions will be an array anyway. (double check shapes of arrays for mask and data)
    return data * mask.astype(int)

if __name__ == "__main__":
    main()