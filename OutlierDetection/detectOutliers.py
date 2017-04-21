import numpy as np
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

COLOR_PALETTE = [    
               "#348ABD",
               "#A60628",
               "#7A68A6",
               "#467821",
               "#CF4457",
               "#188487",
               "#E24A33"
              ]


def plotColumn(dataFrame:pd.DataFrame, x, y):
    plt.xlabel(x)
    plt.ylabel(y)
    xValues = dataFrame[x].values
    yValues = dataFrame[y].values
    plt.plot(xValues, yValues)    

    title = y + ' versus ' + x
    plt.title(title)
    plt.grid(True)
    plt.savefig(title+".png")
    plt.show()

def plotColumns(dataFrame:pd.DataFrame, x, columns):
    f, axarr = plt.subplots(len(columns), sharex=True)
    plt.xlabel(x)
    xValues = dataFrame[x].values
    for  i in range(len(columns)):
        yValues = dataFrame[columns[i]].values
        title = columns[i]
        axarr[i].plot(xValues, yValues)
        axarr[i].set_title(title)        
    figTitle = ' '.join(columns)    
    plt.grid(True)
    plt.savefig(figTitle+".png")
    plt.show()


def get_median_filtered(signal, threshold = 3):    
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal



def __detect_outlier_position_by_fft(signal, threshold_freq=.1, frequency_amplitude=.01):
    fft_of_signal = np.fft.fft(signal)
    outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
    if np.any(np.abs(fft_of_signal[int(threshold_freq):]) > frequency_amplitude):
        index_of_outlier = np.where(signal == outlier)
        return index_of_outlier[0]
    else:
        return None


def detectOutlierFFT(data:pd.DataFrame, xColumn, yColumn, threshold_freq=8,frequency_amplitude=20):
    figTitle = yColumn +" - Outliers using FFT for threshold_freq="+str(threshold_freq)+" and frequency_amplitude="+ str(frequency_amplitude)
    yValues = data[yColumn].values
    xValues = data[xColumn].values

   

    outlier_positions = []
    for ii in range(5, yValues.size, 5):
        outlier_position = __detect_outlier_position_by_fft(yValues[ii-5:ii+5],threshold_freq,frequency_amplitude)
        if outlier_position is not None:
            outlier_positions.append(ii + outlier_position[0] - 5)
    outlier_positions = list(set(outlier_positions))


    plt.figure(figsize=(12, 6));
    plt.xlabel(xColumn)
    plt.ylabel(yColumn)    
    
    plt.scatter(xValues, yValues, c=COLOR_PALETTE[0], label='Original Signal');
    if len(outlier_positions) > 0:
        plt.scatter(xValues[outlier_positions], yValues[np.asanyarray(outlier_positions)], c=COLOR_PALETTE[-1], label='Outliers');
    
    plt.title(figTitle)
    
    plt.legend();
    plt.savefig(figTitle+".png")
    plt.show()
    return outlier_positions
             


    