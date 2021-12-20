from os import listdir
from os.path import isfile, join

import pandas as pd

from skimage import data, io
from skimage.filters import apply_hysteresis_threshold
from skimage.exposure import histogram
import matplotlib.pyplot as plt

import numpy as np

def main():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    images = [input_path + f for f in listdir(input_path) if isfile(join(input_path, f))]
    print(images)
    for i in images:
        img = io.imread(i)
        if len(img.shape) <= 3:
            low, high = determineHysteresisThresholds(img=img, outputPath=output_path, movingAverageFrame=30, cutOffSlope=4)
            normalize_img = img/np.max(img)
            thresholded = hysteresisThresholdingStack(normalize_img, low, high)
            filename = i.split(sep=input_path)[1]
            io.imsave(output_path + filename, thresholded)
        else:
            print("TIME", img.shape)
            threshold_timepoints = []
            print("Number of images: " ,img.shape[0])
            for t in range(img.shape[0]):
                print("Index: ", t)
                low, high = determineHysteresisThresholds(img=img[t], outputPath=output_path, movingAverageFrame=30, cutOffSlope=4)
                normalize_img = img[t]/np.max(img[t])
                threshold_timepoints.append(hysteresisThresholdingStack(normalize_img, low, high))
            thresholded = np.stack(threshold_timepoints)
            print(type(thresholded))
            filename = i.split(sep=input_path)[1]
            io.imsave(output_path + filename, thresholded)
            #NB!!!!!!! Need to Try catch and fix dimensions for FiJi (time + z combined!!)
    return



def hysteresisThresholdingStack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def determineHysteresisThresholds(img, outputPath=None, bins=256, movingAverageFrame=20, cutOffSlope=2, highVal=0.95): #This function is from Rensu's MEL (Make sure to reference)
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
        plt.savefig(outputPath)
        print("Saved histogram")

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))


if __name__ == "__main__":
    main()