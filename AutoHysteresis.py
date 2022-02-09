from os import listdir
from os.path import isfile, join, exists

import pandas as pd

from sklearn.metrics import mean_absolute_error
from skimage import data, io
from skimage.filters import apply_hysteresis_threshold
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

manual_Hysteresis = ["CCCP_1C=0.tif", [[0.1, 0.408], [0.1, 0.25]], "CCCP_1C=1.tif", [[0.116, 0.373], [0.09, 0.22]],"CCCP_2C=0.tif", [[0.107, 0.293], [0.09, 0.2]], "CCCP_2C=1.tif", [[0.09, 0.372], [0.08, 0.15]],"CCCP+Baf_2C=0.tif", [[0.093, 0.279], [0.1, 0.17]], "CCCP+Baf_2C=1.tif", [[0.098, 0.39], [0.1, 0.35]],"Con_1C=0.tif", [[0.197, 0.559], [0.14, 0.18]], "Con_1C=2.tif", [[0.168, 0.308], [0.11, 0.2]],"Con_2C=0.tif", [[0.219, 0.566], [0.19, 0.31]], "Con_2C=2.tif", [[0.137, 0.363], [0.13, 0.23]],"HML+C+B_2C=0.tif", [[0.102, 0.55], [0.14, 0.31]], "HML+C+B_2C=1.tif", [[0.09, 0.253], [0.09, 0.18]],"HML+C+B_2C=2.tif", [[0.114, 0.477], [0.11, 0.31]], "LML+C+B_1C=0.tif", [[0.09, 0.152], [0.05, 0.1]],"LML+C+B_1C=1", [[0.102, 0.232], [0.07, 0.15]], "LML+C+B_1C=2.tif", [[0.034, 0.097], [0.024, 0.1]]]

def main():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Input Data\\"
    output_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Output Data\\"
    threshold_record_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Output Data\\Record.txt"
    thresholded_compare_path = "C:\\RESEARCH\\Mitophagy_data\\Threshold Test Data\\Manual Thresh\\"
    if exists(threshold_record_path):
        record = open(threshold_record_path, 'a')
    else:
        record = open(threshold_record_path, 'w')
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    print(images)
    parameter_variations = [[10, 20],[2, 4]]

    iDict = {}
    for i in images:
        if i in listdir(thresholded_compare_path):
            manual_thresh = image_average(io.imread(input_path+i, ))
            mDict = {}
            for m in parameter_variations[0]:
                cDict = {}
                for c in parameter_variations[1]:
                    HystThreshold = thresholding(i, input_path, output_path, m, c, record)
                    HystThreshold = HystThreshold.astype('uint8')
                    mse_Result = mean_squared_error(manual_thresh/np.max(manual_thresh), HystThreshold)
                    #ssim_Result = ssim(manual_thresh, HystThreshold)
                    cDict[str(c)] = mse_Result
                mDict[str(m)] = cDict
            iDict[i] = mDict
    comparison_results = np.zeros([len(parameter_variations[0]), len(parameter_variations[1])])
    num_files = len(list(iDict))
    for files, values in iDict.items():
        for m_index, m_values in values.items():
            for c_index, c_values in m_values.items():
                comparison_results[parameter_variations[0].index(int(m_index)), parameter_variations[1].index(int(c_index))] += c_values
    comparison_results = comparison_results / num_files
    print(iDict)
    print(comparison_results)
    for m in parameter_variations[0]:
        for c in parameter_variations[1]:
            print("Value for m", m, "and c", c, " = ", comparison_results[parameter_variations[0].index(int(m)), parameter_variations[1].index(int(c))])
    print("C Average", np.mean(comparison_results, axis=0))
    print("M Average", np.mean(comparison_results, axis=1))


    record.close()
    return

def image_average(input_image, parameters, thresh_type):
    thresholded_images = []
    value1 = 0
    value2 = 0
    for p in parameters:
        value1 += p[0]
        value2 += p[1]
        if thresh_type:
            thresholded_images.append(adaptive_threshold_stack(input_image, p[0], p[1]))
        else:
            thresholded_images.append(hysteresisThresholdingStack(input_image, p[0], p[1]).astype(int))
    if thresh_type:
        thresholded_images.append(adaptive_threshold_stack(input_image, value1/len(parameters), value2/len(parameters)))
    else:
        thresholded_images.append(hysteresisThresholdingStack(input_image, value1/len(parameters), value2/len(parameters)).astype(int))
    average_image = np.mean( np.array(thresholded_images), axis=0)
    return np.round(average_image)

def adaptive_threshold_stack(input_image, block_size = 100, constant = -30):
    if block_size % 2 != 1:
        block_size += 1

    thresholded = []
    for i in range(input_image.shape[0]):
        thresholded.append(cv2.adaptiveThreshold(input_image[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)) # ADAPTIVE_THRESH_MEAN_C
    return np.array(thresholded)

def thresholding(i, input_path, output_path, movingAverageFrame, cutOffSlope, record_name):
    img = io.imread(input_path + i)
    print(type(img))
    if len(img.shape) <= 3:
        print(i)
        filename = '.'.join(i.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(
            cutOffSlope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determineHysteresisThresholds(img=img, outputPath=output_path, movingAverageFrame=movingAverageFrame,cutOffSlope=cutOffSlope)
        normalize_img = img / np.max(img)
        thresholded = hysteresisThresholdingStack(normalize_img, low, high)
        #record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        #io.imsave(output_path + filename, thresholded)
        return thresholded

def thresholdingLoop(i, input_path, output_path, movingAverageFrame, cutOffSlope, record_name):
    img = io.imread(i)
    if len(img.shape) <= 3:
        filename = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(
            cutOffSlope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determineHysteresisThresholds(img=img, outputPath=output_path, movingAverageFrame=movingAverageFrame,
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
            #hist_name="Hist of " + fileFirstHalf + "t" + str(t) + fileEndHalf + ".png",
            low, high = determineHysteresisThresholds(img=img[t], outputPath=output_path, movingAverageFrame=movingAverageFrame,
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
    if outputPath != None and hist_name != None:
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.axvline(useIntensityLow, 0, 1, label='Low', color="red")
        plt.axvline((1.0-(1.0-useIntensityLow/bins)/2)*bins, 0, 1, label='High', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        outputPath = outputPath + "\\" + hist_name
        plt.savefig(outputPath)
        print("Saved histogram")

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))


if __name__ == "__main__":
    main()