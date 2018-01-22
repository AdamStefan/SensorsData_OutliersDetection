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

class dataAnalysis(object):    
    def __init__(self, sensorData:du.sensorData):
        self.sensorData = sensorData


    def timeAnalysis(self,printFigures):
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
        

    def __ffwWindowAnalysis(self, windowSize, column, frequenciesBand, printFigure):
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
    
    def fftWindowAnalysis(self, windowSize=10):
        frequenciesBands = [windowSize-2,windowSize-1]
        self.ffwWindowAnalysis(windowSize,'Ext_Tem',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Int_Tem',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Ext_Umi',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Int_Umi',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Int_Pu1',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Int_Pu2',frequenciesBands,True)
        self.ffwWindowAnalysis(windowSize,'Ext_Vvi',frequenciesBands,True)
                           

    #FFT Analysis
    def fftAnalysis(self,printFigure, displayFrequencyAnalisys):                
        frameIntervals = self.sensorData.computeFramesIntervals()

        def printFFT(frequencies,frequenciesNames):
            f, axarr = plt.subplots(len(frequencies), sharex=True)           
            plt.xlabel('frequency unit')            
            for  i in range(len(frequencies)):
                stepPeriod = 15 * 60
                xValues = 1/((np.asarray(list(reversed(range(len(frequencies[i]))))) + 1) * stepPeriod)                
                yValues = frequencies[i]
                arguments = np.argsort(yValues)
                periods = (1/xValues[arguments[0:20]]) / 3600
                print('Dominant periods for ' +  frequenciesNames[i] +  ' are ' +str(periods))
                print('Amplitude in high  frequency for' +  frequenciesNames[i] +  ' is ' +str(yValues[len(yValues)-1]))
                title = frequenciesNames[i]
                axarr[i].plot(xValues, yValues)
                axarr[i].set_title(title)        
            figTitle = 'Frequency domain -' + ' '.join(frequenciesNames)    
            plt.grid(True)
            plt.savefig(figTitle+".png")
            plt.show()

        if displayFrequencyAnalisys:
            for frame in frameIntervals:
                data = frame            
                amplitudesExt_Tem = np.abs(np.fft.fft(data['Ext_Tem'].values))
                amplitutesInt_Tem = np.abs(np.fft.fft(data['Int_Tem'].values))
                amplitudesExt_Umi = np.abs(np.fft.fft(data['Ext_Umi'].values))
                amplitudesInt_Umi = np.abs(np.fft.fft(data['Int_Umi'].values))
                amplitudesInt_Pu1 = np.abs(np.fft.fft(data['Int_Pu1'].values))
                amplitudesInt_Pu2 = np.abs(np.fft.fft(data['Int_Pu2'].values))
                amplitudesExt_Vvi = np.abs(np.fft.fft(data['Ext_Vvi'].values))
                amplitudesToPrint =[amplitudesExt_Tem,amplitutesInt_Tem,amplitudesExt_Umi,amplitudesInt_Umi,amplitudesInt_Pu1,amplitudesInt_Pu2,amplitudesExt_Vvi]
                amplitudesToPrintNames =['Ext_Tem','Int_Tem','Ext_Umi','Int_Umi','Int_Pu1','Int_Pu2','Ext_Vvi']
                printFFT(amplitudesToPrint,amplitudesToPrintNames)

        ######################################################################################
        # consider a window of 1 day
        data = self.sensorData.dataFrame
        windowSize = 10
        outlierDetector = dO.outlierDetector(data,'date')        

        exteTempOutliers = outlierDetector.detectOutlierFFT('Ext_Tem',windowSize -2,25,windowSize= windowSize, printFigure=print)        
        exteUmiOutliers = outlierDetector.detectOutlierFFT('Ext_Umi',windowSize -2,10,windowSize= windowSize,printFigure=print)        
        exteIntUmiOutliers = outlierDetector.detectOutlierFFT('Int_Umi',windowSize -2,500,windowSize= windowSize,printFigure=print)        
        exteIntPuOutliers = outlierDetector.detectOutlierFFT('Int_Pu1',windowSize -2,250,windowSize= windowSize,printFigure=print)        
        exteIntPu2Outliers = outlierDetector.detectOutlierFFT('Int_Pu2',windowSize -2,250,windowSize= windowSize,printFigure=print)
        exteExt_VviOutliers = outlierDetector.detectOutlierFFT('Ext_Vvi',windowSize -2,5,windowSize= windowSize,printFigure=print)

        ######################################################################################

    def fftAnalysis_interval(self,printFigure, displayFrequencyAnalisys, frame_interval_index, columns = None):            
        frameIntervals = self.sensorData.computeFramesIntervals()
        ######################################################################################

        # consider a window of 1 day
        data = frameIntervals[frame_interval_index]
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

        #exteTempOutliers = outlierDetector.detectOutlierFFT('Ext_Tem',windowSize -2,25,windowSize= windowSize, printFigure=print)        
        #exteUmiOutliers = outlierDetector.detectOutlierFFT('Ext_Umi',windowSize -2,10,windowSize= windowSize,printFigure=print)        
        #exteIntUmiOutliers = outlierDetector.detectOutlierFFT('Int_Umi',windowSize -2,500,windowSize= windowSize,printFigure=print)        
        #exteIntPuOutliers = outlierDetector.detectOutlierFFT('Int_Pu1',windowSize -2,250,windowSize= windowSize,printFigure=print)        
        #exteIntPu2Outliers = outlierDetector.detectOutlierFFT('Int_Pu2',windowSize -2,250,windowSize= windowSize,printFigure=print)
        #exteExt_VviOutliers = outlierDetector.detectOutlierFFT('Ext_Vvi',windowSize -2,5,windowSize= windowSize,printFigure=print)
        exteInt_TemOutliers = outlierDetector.detectOutlierFFT('Int_Tem',windowSize -2,50,windowSize= windowSize,printFigure=print)

        if columns is None:
            columns = list(dict.keys())

        handles = [] 
        
        plt.figure(figsize=(12, len(columns)));
        for i in range(len(columns)):
            outliers = outlierDetector.detectOutlierFFT(columns[i],windowSize -2,dict[columns[i]],windowSize= windowSize, printFigure=False)  
            yValues = outlierDetector.data[columns[i]].values
            xValues = outlierDetector.data[outlierDetector.xColumn].values      
            handle_obs = plt.plot(xValues, yValues, c = dO.COLOR_PALETTE[i], label=columns[i])
            handles.append(handle_obs[0])
            if len(outliers) > 0:
                plt.plot(xValues[outliers], yValues[np.asanyarray(outliers)], 'ro');
        figTitle = "- Outliers using FFT -"
        plt.legend(handles=handles, loc="upper left")
        plt.title(figTitle)
        plt.xlabel= outlierDetector.xColumn
        plt.show()


        ######################################################################################    

    #Peculiarity Analysis
    def peculiarityAnalysis(self, print):
        data = self.sensorData.dataFrame
        #######################################################################################
        outlierDetector = dO.outlierDetector(data,'dataTimeStamp')
        results_ExtTemp = outlierDetector.detectOutlierPeculiarity('Ext_Tem',printFigure=print)
        results_Int_Tem = outlierDetector.detectOutlierPeculiarity('Int_Tem',printFigure=print)
        results_Ext_Umi = outlierDetector.detectOutlierPeculiarity('Ext_Umi',printFigure=print)
        results_Int_Umi = outlierDetector.detectOutlierPeculiarity('Int_Umi',printFigure=print)
        results_Int_Pu1 = outlierDetector.detectOutlierPeculiarity('Int_Pu1',printFigure=print)
        results_Int_Pu2 = outlierDetector.detectOutlierPeculiarity('Int_Pu2',printFigure=print)
        results_Ext_Vvi = outlierDetector.detectOutlierPeculiarity('Ext_Vvi',printFigure=print)
        totalOutliers = [results_ExtTemp, results_Int_Tem, results_Ext_Umi, results_Int_Umi, results_Int_Pu2, results_Ext_Vvi]

        dates = []
        data['isOutlier'] = 0

        for outliers in totalOutliers:
            for i in range(len(outliers['anoms'])):
                dates.append(outliers['anoms'].index[i])
                data['isOutlier'][data['date']==outliers['anoms'].index[i]] = 1


        data.to_csv('sensorDataWithOutliers.csv', sep=',', encoding='utf-8') # save the data to a csv file
        ########################################################################################
