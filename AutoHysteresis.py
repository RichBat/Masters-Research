from os import listdir
from os.path import isfile, join, exists

import pandas as pd

from skimage import data, io
from skimage.filters import apply_hysteresis_threshold
from skimage.exposure import histogram
import matplotlib.pyplot as plt

import numpy as np

def main():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    threshold_record_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\Record.txt"
    if exists(threshold_record_path):
        record = open(threshold_record_path, 'a')
    else:
        record = open(threshold_record_path, 'w')
    images = [input_path + f for f in listdir(input_path) if isfile(join(input_path, f))]
    print(images)
    parameter_variations = [[10],[4]]
    for i in images:
        for m in parameter_variations[0]:
            for c in parameter_variations[1]:
                thresholdingLoop(i, input_path, output_path, m, c, record)
    record.close()
    return

def thresholdingLoop(i, input_path, output_path, movingAverageFrame, cutOffSlope, record_name):
    img = io.imread(i)
    if len(img.shape) <= 3:
        filename = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(
            cutOffSlope) + ".tif"
        low, high = determineHysteresisThresholds(img=img, outputPath=output_path, hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png", movingAverageFrame=movingAverageFrame,
                                                  cutOffSlope=cutOffSlope)
        normalize_img = img / np.max(img)
        thresholded = hysteresisThresholdingStack(normalize_img, low, high)
        record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        io.imsave(output_path + filename, thresholded)
    else:
        print("TIME", img.shape)
        threshold_timepoints = []
        print("Number of images: ", img.shape[0])
        fileFirstHalf = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1])
        fileEndHalf = "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif"
        for t in range(img.shape[0]):
            print("Index: ", t)
            low, high = determineHysteresisThresholds(img=img[t], outputPath=output_path, hist_name="Hist of " + fileFirstHalf + "t" + str(t) + fileEndHalf + ".png", movingAverageFrame=movingAverageFrame,
                                                      cutOffSlope=cutOffSlope)
            normalize_img = img[t] / np.max(img[t])
            threshold_timepoints.append(hysteresisThresholdingStack(normalize_img, low, high))
            record_name.write(fileFirstHalf + "t" + str(t) + fileEndHalf + " low: " + str(low * 256) + " high: " + str(high * 256) + "\n")
        thresholded = np.stack(threshold_timepoints)
        #print(type(thresholded))
        filename = fileFirstHalf + fileEndHalf
        io.imsave(output_path + filename, thresholded)
        # NB!!!!!!! Need to Try catch and fix dimensions for FiJi (time + z combined!!)


def hysteresisThresholdingStack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def determineHysteresisThresholds(img, outputPath=None, hist_name=None, bins=256, movingAverageFrame=20, cutOffSlope=2, highVal=0.95): #This function is from Rensu's MEL (Make sure to reference)
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]

    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()

    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0

    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        if movingAverage[0][i-10]/movingAverage[0][i+10] >= cutOffSlope:
              useIntensityLow = i
              print("Low intensity to be used: ", useIntensityLow)
              print("High intensity to be used: ", (1.0-(1.0-useIntensityLow/bins)/2)*bins)

              break

    print(outputPath)
    if outputPath != None:
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.axvline(useIntensityLow, 0, 1, label='Low', color="red")
        plt.axvline((1.0-(1.0-useIntensityLow/bins)/2)*bins, 0, 1, label='High', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        if hist_name != None:
            outputPath = outputPath + "\\" + hist_name
        plt.savefig(outputPath)
        print("Saved histogram")

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))


if __name__ == "__main__":
    main()