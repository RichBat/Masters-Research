from os import listdir
from os.path import isfile, join, exists

import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from skimage import data, io
from skimage.filters import apply_hysteresis_threshold
from skimage.exposure import histogram
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from knee_locator import KneeLocator

manual_Hysteresis = {"CCCP_1C=0.tif": [[0.1, 0.408], [0.1, 0.25]], "CCCP_1C=1.tif": [[0.116, 0.373], [0.09, 0.22]],"CCCP_2C=0.tif": [[0.107, 0.293], [0.09, 0.2]], "CCCP_2C=1.tif": [[0.09, 0.372], [0.08, 0.15]],"CCCP+Baf_2C=0.tif": [[0.093, 0.279], [0.1, 0.17]], "CCCP+Baf_2C=1.tif": [[0.098, 0.39], [0.1, 0.35]],"Con_1C=0.tif": [[0.197, 0.559], [0.14, 0.18]], "Con_1C=2.tif": [[0.168, 0.308], [0.11, 0.2]],"Con_2C=0.tif": [[0.219, 0.566], [0.19, 0.31]], "Con_2C=2.tif": [[0.137, 0.363], [0.13, 0.23]],"HML+C+B_2C=0.tif": [[0.102, 0.55], [0.14, 0.31]], "HML+C+B_2C=1.tif": [[0.09, 0.253], [0.09, 0.18]],"HML+C+B_2C=2.tif": [[0.114, 0.477], [0.11, 0.31]], "LML+C+B_1C=0.tif": [[0.09, 0.152], [0.05, 0.1]],"LML+C+B_1C=1": [[0.102, 0.232], [0.07, 0.15]], "LML+C+B_1C=2.tif": [[0.034, 0.097], [0.024, 0.1]]}

def testing():
    input_path = "C:\\RESEARCH\\Mitophagy_data\\3.Pre-Processed\\"
    images = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    for filename in images:
        img = io.imread(input_path + filename)
        position = testing_knee(img)
        low, high = determine_hysteresis_thresholds(img, moving_average_frame=20, cut_off_slope=4)
        #hist_threshold_differences(filename, low, high, img)
        #position, valid = detectElbows(img)
        valid = True
        if valid:
            print("Sample:", filename, " Threshold:", position)
            print("Low:", low*255, " High:", high*255)
            print(np.array(manual_Hysteresis[filename])*255)
        '''result = image_average(img, manual_Hysteresis[filename], False)
        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(10, 7)
        ax1.imshow(result[int(result.shape[0]/2)])
        plt.show()'''
        '''counts, centers = histogram(img, nbins=256)
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.savefig("C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\Hist.tif")
        print("Saved histogram")'''

def testing_knee(img):
    counts, centers = histogram(img, nbins=256)
    locator = KneeLocator(x=centers[1:], y=counts[1:], curve="convex", direction="decreasing")
    return locator.knee

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

    i_dict = {}
    for i in images:
        if i in listdir(thresholded_compare_path):
            manual_thresh = image_average(io.imread(input_path+i, ))
            m_dict = {}
            for m in parameter_variations[0]:
                c_dict = {}
                for c in parameter_variations[1]:
                    HystThreshold = thresholding(i, input_path, output_path, m, c, record)
                    HystThreshold = HystThreshold.astype('uint8')
                    mse_result = mse(manual_thresh/np.max(manual_thresh), HystThreshold)
                    #ssim_Result = ssim(manual_thresh, HystThreshold)
                    c_dict[str(c)] = mse_result
                m_dict[str(m)] = c_dict
            i_dict[i] = m_dict
    comparison_results = np.zeros([len(parameter_variations[0]), len(parameter_variations[1])])
    num_files = len(list(i_dict))
    for files, values in i_dict.items():
        for m_index, m_values in values.items():
            for c_index, c_values in m_values.items():
                comparison_results[parameter_variations[0].index(int(m_index)), parameter_variations[1].index(int(c_index))] += c_values
    comparison_results = comparison_results / num_files
    print(i_dict)
    print(comparison_results)
    for m in parameter_variations[0]:
        for c in parameter_variations[1]:
            print("Value for m", m, "and c", c, " = ", comparison_results[parameter_variations[0].index(int(m)), parameter_variations[1].index(int(c))])
    print("C Average", np.mean(comparison_results, axis=0))
    print("M Average", np.mean(comparison_results, axis=1))


    record.close()
    return

def hist_threshold_differences(sample_name, calculated_low, calculated_high, image = False):
    print("Low:", calculated_low, " high:", calculated_high)
    manual_values = manual_Hysteresis[sample_name]
    pairs = {}
    for m in range(0, len(manual_values), 1):
        for n in range(m+1, len(manual_values), 1):
            pairs[str(m) + str(n)] = [manual_values[m], manual_values[n]]
    mean_pairs = []
    for k, p in pairs.items():
        distance = math.sqrt((p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2)
        mae = (abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1])) / 2
        pairs[k].append([distance, mae])
        mean_pairs.append([(p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2])
    calc_differences = []
    for manual in manual_values:
        distance = math.sqrt((manual[0] - calculated_low) ** 2 + (manual[1] - calculated_high) ** 2)
        mae = (abs(manual[0] - calculated_low) + abs(manual[1] - calculated_high))/2
        calc_differences.append([distance, mae])
    print(pairs)
    print(calc_differences)
    print(mean_pairs)
    for m_pairs in mean_pairs:
        print("Mean Dist:", math.sqrt((m_pairs[0] - calculated_low) ** 2 + (m_pairs[1] - calculated_high) ** 2))
        print("Mean MAE:", (abs(m_pairs[0] - calculated_low) + abs(m_pairs[1] - calculated_high))/2)
    counts, centers = histogram(image, nbins=256)
    for k, pair in pairs.items():
        # remove 'black'
        start = int(min(pair[0][1], pair[1][1])*255 - 40)
        print(start)
        count = counts[1:]
        center = centers[1:]
        print(center[0])
        print(len(count), len(center))
        plt.figure(figsize=(6, 4))
        plt.plot(center, count, color='black')
        #plt.axvline(pair[0][1]*255, 0, 1, label='FirstHigh', color="red")
        #plt.axvline(pair[1][1]*255, 0, 1, label='SecondHigh', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.show()

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
            thresholded_images.append(hysteresis_thresholding_stack(input_image, p[0]*255, p[1]*255).astype(int))
    if thresh_type:
        thresholded_images.append(adaptive_threshold_stack(input_image, value1/len(parameters), value2/len(parameters)))
    else:
        thresholded_images.append(hysteresis_thresholding_stack(input_image, value1/len(parameters), value2/len(parameters)).astype(int))
    average_image = np.mean(np.array(thresholded_images), axis=0)
    return np.round(average_image)

def adaptive_threshold_stack(input_image, block_size = 100, constant = -30):
    if block_size % 2 != 1:
        block_size += 1

    thresholded = []
    for i in range(input_image.shape[0]):
        thresholded.append(cv2.adaptiveThreshold(input_image[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)) # ADAPTIVE_THRESH_MEAN_C
    return np.array(thresholded)

def thresholding(i, input_path, output_path, moving_average_frame, cut_off_slope, record_name):
    img = io.imread(input_path + i)
    print(type(img))
    if len(img.shape) <= 3:
        print(i)
        filename = '.'.join(i.split(sep=".")[:-1]) + "m" + str(moving_average_frame) + "c" + str(
            cut_off_slope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determine_hysteresis_thresholds(img=img, outputPath=output_path, moving_average_frame=moving_average_frame,cut_off_slope=cut_off_slope)
        normalize_img = img / np.max(img)
        thresholded = hysteresis_thresholding_stack(normalize_img, low, high)
        #record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        #io.imsave(output_path + filename, thresholded)
        return thresholded

def thresholdingLoop(i, input_path, output_path, moving_average_frame, cut_off_slope, record_name):
    img = io.imread(i)
    if len(img.shape) <= 3:
        filename = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1]) + "m" + str(moving_average_frame) + "c" + str(
            cut_off_slope) + ".tif"
        #hist_name="Hist of " + '.'.join(filename.split(sep=".")[:-1]) + ".png"
        low, high = determine_hysteresis_thresholds(img=img, outputPath=output_path, moving_average_frame=moving_average_frame,
                                                  cut_off_slope=cut_off_slope)
        normalize_img = img / np.max(img)
        thresholded = hysteresis_thresholding_stack(normalize_img, low, high)
        record_name.write(filename + " low: " + str(low*256) + " high: " + str(high*256) + "\n")
        #print('.'.join(filename.split(sep=".")[:-1]) + "m" + str(movingAverageFrame) + "c" + str(cutOffSlope) + ".tif")
        print(filename)
        io.imsave(output_path + filename, thresholded)
    else:
        print("TIME", img.shape)
        threshold_timepoints = []
        print("Number of images: ", img.shape[0])
        fileFirstHalf = '.'.join(i.split(sep=input_path)[1].split(sep=".")[:-1])
        fileEndHalf = "m" + str(moving_average_frame) + "c" + str(cut_off_slope) + ".tif"
        for t in range(img.shape[0]):
            print("Index: ", t)
            #hist_name="Hist of " + fileFirstHalf + "t" + str(t) + fileEndHalf + ".png",
            low, high = determine_hysteresis_thresholds(img=img[t], outputPath=output_path, moving_average_frame=moving_average_frame,
                                                      cut_off_slope=cut_off_slope)
            normalize_img = img[t] / np.max(img[t])
            threshold_timepoints.append(hysteresis_thresholding_stack(normalize_img, low, high))
            record_name.write(fileFirstHalf + "t" + str(t) + fileEndHalf + " low: " + str(low * 256) + " high: " + str(high * 256) + "\n")
        thresholded = np.stack(threshold_timepoints)
        #print(type(thresholded))
        filename = fileFirstHalf + fileEndHalf
        io.imsave(output_path + filename, thresholded)
        # NB!!!!!!! Need to Try catch and fix dimensions for FiJi (time + z combined!!)


def hysteresis_thresholding_stack(stack, low=0.25, high=0.7): #Also from Rensu
    return apply_hysteresis_threshold(stack, low, high)

def determine_hysteresis_thresholds(img, outputPath=None, hist_name=None, bins=256, moving_average_frame=20, cut_off_slope=4, highVal=0.95): #This function is from Rensu's MEL (Make sure to reference)
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]

    df = pd.DataFrame(counts)
    movingAverage = df.rolling(moving_average_frame, center=True).mean()

    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0

    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        if movingAverage[0][i-10]/movingAverage[0][i+10] >= cut_off_slope:
              useIntensityLow = i
              print("Low intensity to be used: ", useIntensityLow)
              print("High intensity to be used: ", (1.0-(1.0-useIntensityLow/bins)/4)*bins)

              break

    print(outputPath)
    plt.figure(figsize=(6, 6))
    plt.plot(centers, counts, color='black')
    plt.axvline(useIntensityLow, 0, 1, label='Low', color="red")
    plt.axvline((1.0 - (1.0 - useIntensityLow / bins) / 2) * bins, 0, 1, label='High', color="blue")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("The automatically calculated hysteresis thresholding values")
    plt.tight_layout()
    plt.show()
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

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/4))

def detectElbows(img, movingAverageFrame=20, elbowSize=10):
    #The gradient margin can get big (28%) thus the selected points are not even close to being perpendicular. A line of values between points needs to be generated and then check for interception. If it intercepts between two points then pick the closest or rightmost
    counts, centers = histogram(img, nbins=256)
    #remove black
    counts = counts[1:]
    centers = centers[1:]
    # [intensity, pixel_count]
    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()
    print(movingAverage.min(axis = 0))
    startIntensity = 10
    #print(len(movingAverage[0]))
    diagnose = {}
    distPoints = []
    perpendicularPoints = []
    for i in range(len(movingAverage[0]) - elbowSize - 1, elbowSize + 1, -1):
        rightmost = movingAverage[0][i + elbowSize]
        leftmost = movingAverage[0][i - elbowSize]
        gradient = (rightmost - leftmost)/(2*elbowSize)
        centerIntensity = abs(rightmost-leftmost)/2
        try:
            if gradient > 0:
                best_position = i+1
                best_grad = 0
                first = True
                for p in range(i+1, i+elbowSize, 1):
                    perp_grad = elbowGradientCheck(movingAverage[0][p], centerIntensity, p, i)
                    grad_diff = abs(perp_grad + gradient)
                    if first:
                        first = False
                        best_grad = perp_grad
                        best_position = p
                    else:
                        if grad_diff < abs(best_grad + gradient):
                            best_position = p
                            best_grad = perp_grad
                perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [best_position, movingAverage[0][best_position], best_grad]])
                distance = math.sqrt((best_position-i)**2 + (movingAverage[0][best_position] - centerIntensity)**2)
                distPoints.append([best_position, distance])

            elif gradient < 0:
                best_position = i + 1
                best_grad = 0
                first = True
                for p in range(i-1, i-elbowSize, -1):
                    perp_grad = elbowGradientCheck(movingAverage[0][p], centerIntensity, p, i)
                    grad_diff = abs(perp_grad + gradient)
                    if first:
                        first = False
                        best_grad = perp_grad
                        best_position = p
                    else:
                        if grad_diff < abs(best_grad + gradient):
                            best_position = p
                            best_grad = perp_grad
                perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [best_position, movingAverage[0][best_position], best_grad]])
                distance = math.sqrt((best_position - i) ** 2 + (movingAverage[0][best_position] - centerIntensity) ** 2)
                distPoints.append([best_position, distance])
            else:
                if centerIntensity > movingAverage[0][i]:
                    perpendicularPoints.append([[i, centerIntensity, rightmost, leftmost, gradient], [i, movingAverage[0][i]]])
                    distPoints.append([i, centerIntensity - movingAverage[0][i]])
        except Exception as e:
            print(e)
            print("")

    if distPoints:
        #This is a cheat since I know that distPoints will be of a list type so this should always work
        distances = np.array(distPoints)
        #print(distances[:, 1])
        maximum_values = np.argmax(distances[:, 1])
        #print("Yay", maximum_values)
        #print(distances[maximum_values])
        print(distPoints[maximum_values])
        print("PerpPoints", perpendicularPoints[maximum_values])
        orig_pos = perpendicularPoints[maximum_values][0][0] + elbowSize + 1
        print(movingAverage[0][10:orig_pos])
        movingAverageArray = movingAverage.T.to_numpy()
        print("Average Array:", movingAverageArray[0][10])
        # print(movingAverageArray[0].shape, centers.shape)
        '''plt.figure(figsize=(6, 4))
        plt.plot(centers, movingAverageArray[0], color='black')
        plt.xlabel("Intensity")
        plt.ylabel("AverageCount")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.show()'''
        return distances[maximum_values][0].astype(int), True
    return 0, False





def elbowGradientCheck(intensity1, intensity2, position1, position2):
    if (position1 - position2) == 0:
        print("Divide by zero warning:", position1)
    try:
        return (intensity1 - intensity2) / (position1 - position2)

    except Exception as e:
        print("Ydelta:", (intensity1 - intensity2), "Xdelta:", (position1 - position2))
        return False


if __name__ == "__main__":
    #main()
    testing()