import numpy as np
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pyculiarity import detect_ts
import dataUtility as du
import detectOutliers as dO
from scipy import signal

class data_analysis(object):    
    def __init__(self, sensorData:du.sensorData):
        self.sensorData = sensorData
        self.columns_of_interest = ['Ext_Tem','Int_Tem','Ext_Umi','Int_Umi','Int_Pu1','Int_Pu2','Ext_Vvi']

        
    def time_analysis(self, printFigures):
        if printFigures:
            self.sensorData.plotColumns()
        mean, variance = self.sensorData.computeMeanAndVariance('Ext_Tem')
        print('Ext_Tem - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Int_Tem')
        print('Int_Tem - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Ext_Umi')
        print('Ext_Umi - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Int_Umi')
        print('Int_Umi - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Ext_Vvi')
        print('Ext_Vvi - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Int_Pu1')
        print('Int_Pu1 - mean:'+ str(mean) + ' and variance:' + str(variance))
        mean, variance = self.sensorData.computeMeanAndVariance('Int_Pu2')
        print('Int_Pu2 - mean:'+ str(mean) + ' and variance:' + str(variance))
        

    def __fft_window_analysis(self, windowSize, column, frequenciesBand, printFigure):
        figTitle = "Frequency analysis (window size " + str(windowSize) +" )"
        yValues = self.sensorData.dataFrame[column].values
        ret = []
        for ii in range(int(windowSize/2), yValues.size, int(windowSize/2)):
            windowsAmplitudes = np.abs(np.fft.fft(yValues[ii-int(windowSize/2):ii+int(windowSize/2)]))
            ret.append(windowsAmplitudes)

        ret = np.transpose(np.asarray(ret)) 
        if printFigure:
            f, axarr = plt.subplots(len(frequenciesBand), sharex=True)           
            plt.xlabel('frequency unit')
            for i in range(len(frequenciesBand)):
                frequencyBand = frequenciesBand[i]
                stepPeriod = 1/((len(ret)-frequencyBand) * 15 * 60)
                values, base = np.histogram(ret[frequencyBand])
                xValues = np.asarray(list(range(len(ret[frequencyBand]))))
                yValues = ret[frequencyBand]
                title = 'Frequency ' + str(stepPeriod)
                axarr[i].hist (yValues)
                axarr[i].set_title(title)
            plt.grid(True)
            plt.savefig(figTitle+".png")
            plt.show()                
    
    def fft_window_Analysis(self, windowSize=10, columns_of_interest=[]):
        if columns_of_interest is None or len(columns_of_interest) == 0:
            columns_of_interest = self.columns_of_interest
        frequenciesBands = [windowSize-2,windowSize-1]
        for index in range(len(columns_of_interest)):
            self.__fft_window_analysis(windowSize, columns_of_interest[index], frequenciesBands, True)                
                           

    #FFT Analysis
    def fft_analysis(self, printFigure, displayFrequencyAnalisys, columns=[]):                
        frameIntervals = self.sensorData.computeFramesIntervals()

        if (columns is None) or len(columns) == 0:                    
            columns = self.columns_of_interest

        def print_fft(frequencies, frequencies_names):
            f, axarr = plt.subplots(len(frequencies), sharex=True)           
            plt.xlabel('frequency unit')            
            for  i in range(len(frequencies)):
                stepPeriod = 15 * 60
                xValues = 1/((np.asarray(list(reversed(range(len(frequencies[i]))))) + 1) * stepPeriod)                
                yValues = frequencies[i]
                arguments = np.argsort(yValues)
                periods = (1/xValues[arguments[0:20]]) / 3600
                print('Dominant periods for ' +  frequencies_names[i] +  ' are ' +str(periods))
                print('Amplitude in high  frequency for' +  frequencies_names[i] +  ' is ' +str(yValues[len(yValues)-1]))
                title = frequencies_names[i]
                axarr[i].plot(xValues, yValues)
                axarr[i].set_title(title)        
            figTitle = 'Frequency domain -' + ' '.join(frequencies_names)    
            plt.grid(True)
            plt.savefig(figTitle+".png")
            plt.show()

        if displayFrequencyAnalisys:
            for frame in frameIntervals:
                data = frame
                amplitudes_to_print = []
                amplitudesToPrintNames = []
                for column_name in columns:                
                    amplitudes= np.abs(np.fft.fft(data[column_name].values))
                    amplitudes_to_print.append(amplitudes)                    
                print_fft(amplitudes_to_print,columns)                                

        ######################################################################################
        # consider a window of 1 day
        data = self.sensorData.dataFrame
        windowSize = 10
        outlierDetector = dO.outlierDetector(data,'date')        
        
        dict_thresholds = {
                                'Ext_Tem':25,
                                'Ext_Umi':10,
                                'Int_Umi':500,
                                'Int_Pu1':250,
                                'Int_Pu2':250,
                                'Ext_Vvi':5,
                                'Int_Tem':50
                          }        

        outlier_positions = []        
        for column_index in range(len(columns)):
            outliers = outlierDetector.detect_outlier_fft(columns[column_index], windowSize -2, dict_thresholds[columns[column_index]], windowSize= windowSize, printFigure=print)        
            for outlier_index in range(len(outliers)):
                outlier_item = {}
                outlier_item["Date"] = data['date'].values[outliers[outlier_index]]
                outlier_item["Feature"] = columns[column_index]
                outlier_positions.append(outlier_item)

        if len(outlier_positions)>0:
            return pd.DataFrame(outlier_positions)        

        

    def fft_analysis_interval(self, printFigure, displayFrequencyAnalisys, frame_interval, columns = None):                           
        # consider a window of 1 day
        data = frame_interval
        windowSize = 10
        outlierDetector = dO.outlierDetector(data,'date')
        dict = {
                'Ext_Tem':25,
                'Ext_Umi':10,
                'Int_Umi':500,
                'Int_Pu1':250,
                'Int_Pu2':250,
                'Ext_Vvi':5,
                'Int_Tem':50
                }                

        if columns is None:
            columns = list(dict.keys())

        handles = [] 
        
        plt.figure(figsize=(12, len(columns)));
        outlier_positions = []

        for i in range(len(columns)):
            outliers = outlierDetector.detect_outlier_fft(columns[i],windowSize -2,dict[columns[i]],windowSize= windowSize, printFigure=False)  
            yValues = outlierDetector.data[columns[i]].values
            xValues = outlierDetector.data[outlierDetector.xColumn].values      
            handle_obs = plt.plot(xValues, yValues, c = dO.COLOR_PALETTE[i], label=columns[i])
            handles.append(handle_obs[0])
            if len(outliers) > 0:
                plt.plot(xValues[outliers], yValues[np.asanyarray(outliers)], 'ro');

                for outlier_index in range(len(outliers)):
                    outlier_item = {}
                    outlier_item["Date"] = data['date'].values[outliers[outlier_index]]
                    outlier_item["Feature"] = columns[i]
                    outlier_positions.append(outlier_item)


        figTitle = "- Outliers using FFT -"
        plt.legend(handles=handles, loc="upper left")
        plt.title(figTitle)
        plt.xlabel= outlierDetector.xColumn
        plt.show()

        if len(outlier_positions)>0:
            return pd.DataFrame(outlier_positions)

        ######################################################################################    

    #Peculiarity Analysis
    def peculiarity_analysis(self, frame, columns, print):
        data = frame
        outlier_positions = []
        totalOutliers = []
        
        #######################################################################################
        
        for column in columns:
            outlierDetector = dO.outlierDetector(data,'dataTimeStamp')
            outliers = outlierDetector.detect_outlier_peculiarity(frame, column, printFigure=print)
            totalOutliers.append(outliers)                   

        dates = []
        data['isOutlier'] = 0

        for feature_outliers_index in range(len(totalOutliers)):
            outliers = totalOutliers[feature_outliers_index]
            for i in range(len(outliers['anoms'])):

                dates.append(outliers['anoms'].index[i])
                itemindex = np.where(data['date'].values==np.datetime64(outliers['anoms'].index[i]))
                data['isOutlier'][data['date']==outliers['anoms'].index[i]] = 1

                outlier_item = {}
                outlier_item["Date"] = frame['date'].values[itemindex[0][0]]
                outlier_item["Feature"] = columns[feature_outliers_index]
                outlier_positions.append(outlier_item)    


        data.to_csv('sensorDataWithOutliers.csv', sep=',', encoding='utf-8') # save the data to a csv file
        ########################################################################################

        if len(outlier_positions)>0:
            return pd.DataFrame(outlier_positions)
