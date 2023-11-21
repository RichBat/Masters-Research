from os import listdir
from os.path import isfile, join

from skimage import data, io
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_isodata, threshold_mean, gaussian, apply_hysteresis_threshold,  unsharp_mask
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from skimage.morphology import white_tophat
from skimage.morphology import disk, square, ball

import pandas as pd

import TiffMetadata
from scipy import ndimage

# from CZI_Processor import cziFile
import tifffile

import cv2

import numpy as np

from skimage.exposure import histogram

import matplotlib.pyplot as plt

import pylab


def plotRGBhist(img, bins=256, remove_zero=False, remove_max=False):
    if len(img.shape) != 3:
        print("The provided image must be (X, Y, C)")
        return

    startIndex = 0
    if remove_zero:
        startIndex = 1

    plt.figure()  # this is necessary to ensure no existing figure is overwritten
    r_counts, r_centers = histogram(img[:,:,0], nbins=bins)
    g_counts, g_centers = histogram(img[:, :, 1], nbins=bins)
    b_counts, b_centers = histogram(img[:, :, 2], nbins=bins)

    if remove_max:
        plt.plot(r_centers[startIndex:-1], r_counts[startIndex:-1], color='red')
        plt.plot(g_centers[startIndex:-1], g_counts[startIndex:-1], color='green')
        plt.plot(b_centers[startIndex:-1], b_counts[startIndex:-1], color='blue')
    else:
        plt.plot(r_centers[startIndex:], r_counts[startIndex:], color='red')
        plt.plot(g_centers[startIndex:], g_counts[startIndex:], color='green')
        plt.plot(b_centers[startIndex:], b_counts[startIndex:], color='blue')

    plt.show()

def plotGrayhist(img, bins=256, remove_zero=False, remove_max=False):
    startIndex = 0
    if remove_zero:
        startIndex = 1

    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.reshape(img, (img.shape[0], img.shape[1]))
    elif len(img.shape) != 2:
        print("The provided image must be (X, Y)")
        return

    plt.figure()  # this is necessary to ensure no existing figure is overwritten
    counts, centers = histogram(img, nbins=bins)

    if remove_max:
        plt.plot(centers[startIndex:-1], counts[startIndex:-1], color='black')
    else:
        plt.plot(centers[startIndex:], counts[startIndex:], color='black')

    plt.show()

def plotImageHist(img, bins=256, remove_zero=False, remove_max=False):
    imgShape = img.shape;

    if len(imgShape) == 2:  # gray
        plotGrayhist(img, bins, remove_zero, remove_max)
    elif len(imgShape) == 3:  # rgb
        plotRGBhist(img, bins, remove_zero, remove_max)
    else:
        print("img does not have correct shape. Expected (X, Y, C) or (X, Y)")
        return

def plotStackHist(stack, bins=256, remove_zero=False, remove_max=False):
    stackShape = stack.shape;

    if len(stackShape) == 3: # gray
        plotGrayhist(np.reshape(stack, (stackShape[1], stackShape[2] * stackShape[0])), bins, remove_zero, remove_max)
    elif len(stackShape) == 4: # rgb
        plotRGBhist(np.reshape(stack, (stackShape[1], stackShape[2] * stackShape[0], stackShape[3])), bins, remove_zero, remove_max)
    else:
        print("stack does not have correct shape. Expected (Sl, X, Y, C) or (Sl, X, Y)")
        return

def loadTimelapseTif(path, scaleFactor=1, pad=False):
    stackTimelapsePath = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith("tif") or f.endswith("tiff"))]
    stackTimelapsePath.sort()

    timelapse = [padStack3D(rescaleStackXY(io.imread(path + f), scaleFactor)) if pad else io.imread(path + f) for f in stackTimelapsePath]
    metadata = [TiffMetadata.metadata(path + f) for f in stackTimelapsePath]

    return timelapse, metadata

def loadTifStack(filename, scaleFactor=1, pad=False):
    stack = padStack3D(rescaleStackXY(io.imread(filename), scaleFactor)) if pad else io.imread(filename)
    metadata = TiffMetadata.metadata(filename)

    return stack, metadata

def loadGenericImage(filename):
    return io.imread(filename)

def saveGenericImage(filename, image):
    io.imsave(filename, image)

def saveTifStack(filename, imageArray):
    if(not str(filename).lower().endswith('tif') and not str(filename).lower().endswith('tiff')):
        filename += '.tif'
    io.imsave(filename, imageArray)

def loadCZIFile(filename):
    return cziFile(filename)

def padImageTo3D(image):
    if(image.shape[0] == 1):
        return padStack3D(image)
    else:
        return padStack3D(np.expand_dims(image, axis=0))

def padStack3D(stack):
    newStack = []
    # TODO: Check if I maybe need to add padding of 3 (As before) instead of only 1
    blankSlice = cv2.copyMakeBorder(np.zeros_like(stack[0]), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    newStack.append(blankSlice)
    for im in stack:
        newStack.append(cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    newStack.append(blankSlice)

    return np.stack(newStack)

def padStackXY(stack, paddingWidth=10):
    newStack = []
    pw = int(paddingWidth)

    for im in stack:
        newStack.append(cv2.copyMakeBorder(im, pw, pw, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0)))

    return np.stack(newStack)

def padStackXY_depthLast(stack, paddingWidth=10):
    newStack = []
    pw = int(paddingWidth)

    for i in range(0, stack.shape[2]):
        im = stack[:, :, (stack.shape[2]-1) - i]
        newStack.append(cv2.copyMakeBorder(im, pw, pw, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0)))

    return np.stack(newStack)

def binarizeStack(stack, method='otsu'):
    if method.lower() == 'otsu':
        thresh = threshold_otsu(stack)
    elif method.lower() == 'li':
        thresh = threshold_li(stack)
    elif method.lower() == 'mean':
        thresh = threshold_mean(stack)
    elif method.lower() == 'yen':
        thresh = threshold_yen(stack)
    elif method.lower() == 'isodata':
        thresh = threshold_isodata(stack)
    else:
        print("Error method {} is not valid for binarizeStack. Defaulted to Otsu.".format(method))
        thresh = threshold_otsu(stack)

    return (stack > thresh)

def hysteresisThresholdingStack(stack, low=0.25, high=0.7):
    return apply_hysteresis_threshold(stack, low, high) > 0

def determineHysteresisThresholds(img, outputPath=None, bins=256, movingAverageFrame=20, cutOffSlope=2, highVal=0.95):
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]

    # normalise
    # counts = counts/np.sum(counts)

    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()


    # first derivitive
    # gradient = []
    # gradient2 = []
    # for i in range(0,len(counts)-1):
    #     gradient.append(counts[i+1] - counts[i])
    
    # gradient.append(0)
    # df = pd.DataFrame(gradient)
    # movingAverage = df.rolling(movingAverageFrame, center=True).mean()


    #print(movingAverage)

    # calculate second derivitive
    # for i in range(0,len(movingAverage[0])-1):
    #     gradient2.append(movingAverage[0][i+1] - movingAverage[0][i])
    
    # gradient2.append(0)
    # plt.plot(centers[startIndex:], gradient[startIndex:], color='black')
    # plt.show()

    # df = pd.DataFrame(gradient2)
    # movingAverage = df.rolling(movingAverageFrame, center=True).mean()
    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0
    #for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        # print(i,": ", movingAverage[0][i-10]/movingAverage[0][i+10] )
        #if movingAverage[0][i] <= mEdges:
        if movingAverage[0][i-10]/movingAverage[0][i+10] >= cutOffSlope:
              useIntensityLow = i
              print("Low intensity to be used: ", useIntensityLow/bins)
              print("High intensity to be used: ", (1.0-(1.0-useIntensityLow/bins)/2))
              #print("High intensity to be used: ", highVal)
              break  

    print(outputPath)
    if outputPath != None:
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black') #movingAverage[0]
        plt.axvline(useIntensityLow/bins, 0, 1, label='Low', color="red")
        plt.axvline((1.0-(1.0-useIntensityLow/bins)/2), 0, 1, label='High', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        #plt.axvline(highVal, 0, 1, label='High')
        plt.savefig(outputPath)
        print("Saved histogram")

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))
    #return (useIntensityLow/bins, highVal)

def contrastStretch(stack, l=2, h=100):
    p2, p98 = np.percentile(stack, (l, h))
    outStack = rescale_intensity(stack, in_range=(p2, p98))
    return outStack

def contrastStretchSliceInStack(stack, l=2, h=100):
    #per slice contrast stretching
    slCount = 0
    for sl in stack:
        p_min = np.percentile(sl, l)
        p_max = np.percentile(sl, h)
        stack[slCount] = rescale_intensity(sl, in_range=(p_min, p_max))
        slCount += 1

    return stack

def preprocess(stack, scaleFactor, percentageSaturate=0.003, scaleIntensityFactor=4, sigma2D=1.0, radius=3, amount=3, tophatBallSize=5):
    scaled = rescaleStackXY(stack, scaleFactor=scaleFactor)

    for sl in range(0,scaled.shape[0]):
        scaled[sl] = gaussian(scaled[sl], sigma2D)

    # sharpened = unsharp_mask(scaled, radius, amount)

    normalized = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    counts, centers = histogram(normalized, nbins=256)
    maxVal = 250
    while counts[-1]/np.sum(counts) < percentageSaturate:
        normalized = cv2.normalize(scaled, None, 0, maxVal, cv2.NORM_MINMAX, cv2.CV_8UC1)
        counts, centers = histogram(normalized, nbins=256)
        maxVal += 50
        # print("Max Val = ", maxVal)
        # print(counts[-1]/np.sum(counts))

    print("NORMALIZATION Max Val = ", maxVal)
    #normalized = cv2.normalize(scaled, None, 0, 255*scaleIntensityFactor, cv2.NORM_MINMAX, cv2.CV_8UC1)
    normalized = rescaleStackXY(normalized, scaleFactor=1)
    
    
    # for i in range(0,5):
    #     sharpened = unsharp_mask(normalized, radius, i)
    #     plt.imshow(sharpened[2])
    #     plt.show()

    # Top-hat
    # imageStack = white_tophat(normalized, ball(tophatBallSize))

    return normalized

def unsharpMask(stack, radius=2, amount=2):
    # for i in range(1,5):
    #     sharpened = unsharp_mask(stack, i, amount)
    #     plt.imshow(sharpened[2])
    #     plt.show()
    return unsharp_mask(stack, radius, amount)

def rescaleStackXY(stack, scaleFactor=2, order=1):
    # if scaleFactor == 1:
    #     return stack

    imList = []
    for im in stack:
        imList.append(rescale(im, scaleFactor, preserve_range=True, order=order))

    return np.array(imList)

def rescaleStackXY_depthLast(stack, scaleFactor=2):
    # if scaleFactor == 1:
    #     return stack

    imList = []
    for i in range(0, stack.shape[2]):
        im = stack[:, :, i]
        imList.append(rescale(im, scaleFactor))

    return np.moveaxis(np.array(imList),0,2)

def rescaleStackXY_RGB(stack, scaleFactor=2):
    if scaleFactor == 1:
        return stack

    imList = []
    for im in stack:
        imList.append(rescale(im, scaleFactor, multichannel=True))

    return np.array(imList)

def rescaleImageRGB(image, scaleFactor=2):
    if scaleFactor == 1:
        return image

    return rescale(image, scaleFactor, multichannel=True)

def stackToMIP(stack):
    return np.max(stack, axis=0)

def saveCroppedImagePanel(Frame1, Frame2, EventsFrameRGB, cropXStart, cropXWidth, cropYStart, cropYHeight, outputPath=None):
    frame1StackRGB = np.stack((Frame1,Frame1,Frame1), axis=-1)
    frame2StackRGB = np.stack((Frame2,Frame2,Frame2), axis=-1)

    miniPanel = np.hstack((
        np.max(frame1StackRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],
        np.ones((cropYHeight,2,3)),
        np.max(frame2StackRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],
        np.ones((cropYHeight,2,3)),
        np.max(EventsFrameRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],)
        )
    
    if type(outputPath) != type(outputPath):
        io.imsave(outputPath, (miniPanel*255).astype(np.uint8))

    return miniPanel



def update(val):
    global z
    global low
    global high
    global stack
    global ax2
    global ax
    global centers
    global gradient
    global gradient2
    global startIndex
    global movingAverage
    thresholded = hysteresisThresholdingStack(stack, low.val, high.val)
    #thresholded = binarizeStack(stack)
    stackRGB = np.stack((stack, stack, stack), axis=-1)
    thresholdedRGB = np.stack((thresholded, np.zeros_like(thresholded),np.zeros_like(thresholded)), axis=-1)
    finalStack = stackRGB *0.5 + thresholdedRGB *0.5
    #ax2.imshow(stack[int(z.val)])
    ax2.imshow(finalStack[int(z.val)])

    ax.cla()
    ax.plot(centers[startIndex:], movingAverage[0][startIndex:], color='black')
    ax.axvline(low.val, 0, 1, label='Low')
    ax.axvline(high.val, 0, 1, label='High')

def chooseHysteresisParams(img, bins=256, remove_zero=False, remove_max=False):
    global z
    global low
    global high
    global stack
    global ax2
    global ax
    global centers
    global gradient
    global gradient2
    global startIndex
    global movingAverage
    ax = None
    ax2 = None
    stack = img
    startIndex = 0
    if remove_zero:
        startIndex = 1
    
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]

    df = pd.DataFrame(counts[startIndex:])
    movingAverage = df.rolling(20, center=True).mean()

    (lowInit, highInit) = determineHysteresisThresholds(img)

    ax = plt.subplot(121)
    plt.subplots_adjust(left=0.15, bottom=0.4)    

    # ax.grid(False)
    # ax.title('Choose hysteresis thresholds')
    # ax.xlabel('Intensity')
    # ax.ylabel('concentration')

    axcolor = 'lightgoldenrodyellow'
    axLow = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    axHigh = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    axZ = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)

    low = pylab.Slider(axLow, 'Low', 0.0, 1.0, valinit=lowInit)
    high = pylab.Slider(axHigh, 'High', 0.0, 1.0, valinit=highInit)
    z = pylab.Slider(axZ, 'z', 0, img.shape[0] - 1, valinit=2, valstep = 1)

    ax.plot(centers[startIndex:], movingAverage[0][startIndex:], color='black')
    ax.axvline(low.val, 0, 1, label='Low')
    ax.axvline(high.val, 0, 1, label='High')

    ax2 = plt.subplot(122)
    thresholded = hysteresisThresholdingStack(stack, low.val, high.val)
    #thresholded = binarizeStack(stack)
    stackRGB = np.stack((stack, stack, stack), axis=-1)
    thresholdedRGB = np.stack((thresholded, np.zeros_like(thresholded),np.zeros_like(thresholded)), axis=-1)
    finalStack = stackRGB *0.6 + thresholdedRGB *0.4
    #ax2.imshow(stack[int(z.val)])
    ax2.imshow(finalStack[int(z.val)])

    low.on_changed(update)
    high.on_changed(update)
    z.on_changed(update)

    plt.show()

