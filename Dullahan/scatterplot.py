# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:36:45 2018

@author: Rensu Theart
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.colors as colors
from skimage import data, io
import math

from os import listdir
from os.path import isfile, join
import os

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = "18"
class scatterGen:
    def __init__(self, input_path, ch1, ch2, th1, th2):
        path = input_path #"C:\\Users\\richy\\Desktop\\Python File Backups\\"
        ch1File = ch1 #"GreenSphere.tif"
        ch2File = ch2 #"RedSphere.tif"

        #Parameters for Scatter plot
        self.ch1Stack = io.imread(path + ch1File)
        self.ch2Stack = io.imread(path + ch2File)
        self.thresh1 = th1
        self.thresh2 = th2
        self.Imax = 255
        self.texSize = 255
        self.percToInclude = 0.10
        self.PrepData()

    def PrepData(self):
        #CODE TAKEN/ADAPTED FROM RACC.py
        #Setting up scatter data, averages, max intensity and filtering stacks
        maxIntensity1 = np.max(self.ch1Stack)
        maxIntensity2 = np.max(self.ch2Stack)
        maxInt = np.max((maxIntensity1, maxIntensity2))
        print(self.ch1Stack.shape)
        if(len(self.ch1Stack.shape) == 4):
            self.ch1Stack = np.mean(self.ch1Stack, axis=3, dtype=int)
            self.ch2Stack = np.mean(self.ch2Stack, axis=3, dtype=int)



        self.Z = self.ch1Stack.shape[0]
        self.W = self.ch1Stack.shape[1]
        self.H = self.ch1Stack.shape[2]

        #print(np.average(self.testScatter))
        #print("Max of scatter ", np.max(self.testScatter))
        if (maxInt > 255):
            print("Max intensity greater than 255 ({}), adjusted".format(maxInt))
            self.ch1Stack = self.ch1Stack / maxInt * 255
            self.ch2Stack = self.ch2Stack / maxInt * 255

    def generateScatter(self):
        #Calculating Averages and Covariances
        threshCh1 = self.thresh1
        threshCh2 = self.thresh2

        testScatter = np.zeros(shape=(256, 256), dtype=int)

        valuesAboveThreshCh1 = self.ch1Stack[np.where(self.ch1Stack >= threshCh1)]
        valuesAboveThreshCh2 = self.ch2Stack[np.where(self.ch2Stack >= threshCh2)]

        averageCh1 = np.average(valuesAboveThreshCh1)
        averageCh2 = np.average(valuesAboveThreshCh2)

        filteredCh1Stack = np.copy(self.ch1Stack)
        filteredCh2Stack = np.copy(self.ch2Stack)
        filteredCh1Stack[filteredCh1Stack < threshCh1] = 0
        filteredCh2Stack[filteredCh2Stack < threshCh2] = 0

        for z in range(self.Z):
            for w in range(self.W):
                for h in range(self.H):
                    try:
                        testScatter[filteredCh2Stack[z,w,h], filteredCh1Stack[z,w,h]] += 1
                    except Exception as e:
                        print("********************************************************")
                        print(z, w, h)
                        print(filteredCh1Stack[z,w,h], filteredCh2Stack[z,w,h])
                        print(e)
                        print("--------------------------------------------------------")


        filteredCh1Stack = filteredCh1Stack.ravel()
        filteredCh2Stack = filteredCh2Stack.ravel()

        print("Flattened sizes Ch1 & Ch2 ", filteredCh1Stack.shape, filteredCh2Stack.shape)
        #scatter = np.column_stack((filteredCh1Stack, filteredCh2Stack))
        #np.savetxt(path + "foo.csv", scatter, delimiter=",")


        covariance = np.cov(filteredCh1Stack, filteredCh2Stack)
        varXX = covariance[0, 0]
        varYY = covariance[1, 1]
        varXY = covariance[0, 1]

        #print(testScatter[45:60, 180:210])

        # Calculate B0 and B1

        lamb = 1  # special case of Deming regression
        val = lamb * varYY - varXX

        B0 = 0
        B1 = 0
        print("Covariance Value: ", varXY)
        if (varXY < 0):
            print("\nThe covariance is negative")
            B1 = (val - math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)
        else:
            B1 = (val + math.sqrt(val * val + 4 * lamb * varXY * varXY)) / (2 * lamb * varXY)

        B0 = averageCh2 - B1 * averageCh1
        print("Intercept: ", B0, " Slope: ", B1, " Covariance: ", varXY)
        #Calculate p1

        p1 = np.zeros(2)

        # For P1
        if (B0 >= self.Imax * (1 - B1)):
            p1[0] = (self.Imax - B0) / B1
            p1[1] = self.Imax
        elif (B0 < self.Imax * (1 - B1)):
            p1[0] = self.Imax
            p1[1] = self.Imax * B1 + B0

        '''
        # SET THESE VALUES
        thresh1 = 1
        thresh2 = 1
        
        m =0.1095179                      
        c =0.3612732          *255
        p_end =[1, 0.4707911]       
        distThresh = 159
        
        #plt.title('Partially overlapping sphere-sphere (MIP)\n', fontsize=24)
        '''

        totalVoxelCount = 0
        colorMapFrequencyX = np.zeros(self.texSize)
        colorMapFrequencyY = np.zeros(self.texSize)

        overlappingSection = np.multiply(np.clip(filteredCh1Stack, 0, 1), np.clip(filteredCh2Stack, 0, 1)) * self.Imax
        reducedCh1Stack = filteredCh1Stack[overlappingSection > 0]
        reducedCh2Stack = filteredCh2Stack[overlappingSection > 0]
        print("\nFull size was {} reduced colocalized size is {}. Remaining percentage {}%".format(filteredCh1Stack.shape,
                                                                                                   reducedCh1Stack.shape,
                                                                                                   reducedCh1Stack.shape[0] /
                                                                                                   filteredCh1Stack.shape[
                                                                                                       0] * 100))

        totalVoxelCount = reducedCh2Stack.shape[0]

        # CALCULATE distance threshold (variant of binary search)

        distanceCount = 0
        dThresh = 0
        tryCount = 0
        dMin = 0
        dMax = self.Imax
        dThresh = dMin + (dMax - dMin) / 2.0

        #Assigning parameters calculated in RACC formulae to the original Scatter.py

        distThresh = dThresh

        xMean = averageCh1
        yMean = averageCh2

        print ("xMean = " + str(xMean) + " yMean = " + str(yMean))


        #imgplot = plt.imshow(scatter)
        p_end = p1
        m = B1
        c = B0
        #Scatter.py CALCULATIONS
        # channel 1 threshold
        x1 = [threshCh1,threshCh1]
        y1 = [0,255]

        # channel 2 threshold
        x2 = [0,255]
        y2 = [threshCh2,threshCh2]

        #line = plt.plot(x1, y1, '--', linewidth=2.0, color="white")
        #line = plt.plot(x2, y2, '--', linewidth=2.0, color="white")

        # regression line
        x_r = [0,255]
        y_r = [c,x_r[1]*m + c]

        line = plt.plot(x_r, y_r, '-', linewidth=1.0, color="red")

        # max line
        m_max = -1/m
        c_max = p_end[1] - p_end[0]*m_max

        x_max = [0,255]
        y_max = [c_max,x_max[1]*m_max + c_max]

        #line = plt.plot(x_max, y_max, '-', linewidth=2.0, color="orange")

        #distance lines
        theta = np.arctan(m)
        c_dist1 = c + distThresh/np.cos(theta)
        c_dist2 = c - distThresh/np.cos(theta)

        x_dist = [0,255]
        y_dist1 = [c_dist1,x_max[1]*m + c_dist1]
        y_dist2 = [c_dist2,x_max[1]*m + c_dist2]

        #line = plt.plot(x_dist, y_dist1, '-', linewidth=2.0, color="white")
        #line = plt.plot(x_dist, y_dist2, '-', linewidth=2.0, color="white")

        plt.plot([xMean], [yMean],".",markersize=20, color="red")
        imgplot = plt.imshow(testScatter, norm=matplotlib.colors.LogNorm(), interpolation='none')
        plt.gca().invert_yaxis()
        imgplot.set_cmap('viridis') #'jet'  'plasma' 'magma'

        plt.xlabel('Channel 1 intensity', fontsize=22)
        plt.ylabel('Channel 2 intensity', fontsize=22)


        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Frequency", fontsize=22)

        #plt.savefig(pathToScatter+path + ".png")
        plt.show()

    def justCov(self, th1, th2):
        #Calculating Averages and Covariances
        threshCh1 = th1
        threshCh2 = th2

        testScatter = np.zeros(shape=(256, 256), dtype=int)


        valuesAboveThreshCh1 = self.ch1Stack[np.where(self.ch1Stack >= threshCh1)]
        valuesAboveThreshCh2 = self.ch2Stack[np.where(self.ch2Stack >= threshCh2)]

        averageCh1 = np.average(valuesAboveThreshCh1)
        averageCh2 = np.average(valuesAboveThreshCh2)

        filteredCh1Stack = np.copy(self.ch1Stack)
        filteredCh2Stack = np.copy(self.ch2Stack)
        filteredCh1Stack[filteredCh1Stack < threshCh1] = 0
        filteredCh2Stack[filteredCh2Stack < threshCh2] = 0

        for z in range(self.Z):
            for w in range(self.W):
                for h in range(self.H):
                    try:
                        testScatter[filteredCh1Stack[z,w,h], filteredCh2Stack[z,w,h]] += 1
                    except Exception as e:
                        print("********************************************************")
                        print(z, w, h)
                        print(filteredCh1Stack[z,w,h], filteredCh2Stack[z,w,h])
                        print(e)
                        print("--------------------------------------------------------")


        filteredCh1Stack = filteredCh1Stack.ravel()
        filteredCh2Stack = filteredCh2Stack.ravel()

        print("Flattened sizes Ch1 & Ch2 ", filteredCh1Stack.shape, filteredCh2Stack.shape)
        #scatter = np.column_stack((filteredCh1Stack, filteredCh2Stack))
        #np.savetxt(path + "foo.csv", scatter, delimiter=",")


        covariance = np.cov(filteredCh1Stack, filteredCh2Stack)
        varXY = covariance[0, 1]
        return varXY

    def searchForPos(self, stepSize=5, limitSize=255):
        better_cov = 0
        cov_calculated = False
        bestXY = [0, 0]
        improved_cov = False
        for thx in range(0, limitSize, stepSize):
            for thy in range(0, limitSize, stepSize):
                cov_calculated = self.justCov(th1=thx, th2=thy)
                if(cov_calculated > 0):
                    return [thx, thy], cov_calculated, True
                if(cov_calculated > better_cov and cov_calculated):
                    better_cov = cov_calculated
                    bestXY[0] = thx
                    bestXY[1] = thy
                    improved_cov = True
                if(not cov_calculated):
                    better_cov = cov_calculated
                    cov_calculated = True
        return bestXY, better_cov, improved_cov

if __name__ == '__main__':
    path = "C:\\Users\\19116683\\Desktop\\Deconvolved Testing\\"
    ch1File = "CCCP+B_3C=0.tif"
    ch2File = "CCCP+B_3C=2.tif"
    thresh1 = 10
    thresh2 = 10
    scatter = scatterGen(input_path=path, ch1=ch1File, ch2=ch2File, th1=thresh1, th2=thresh2)
    #threshValues, cov, improved = scatter.searchForPos(stepSize=10, limitSize=100)
    #print("Search Results: ", threshValues, cov, improved)
    scatter.generateScatter()