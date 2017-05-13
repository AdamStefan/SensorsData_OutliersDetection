import json
import os
import sys
import pandas as pd
import detectOutliers as dO
import matplotlib.pyplot as plt
import numpy as np
from pyculiarity import detect_ts
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import dataUtility 


dataPath = "..\\data2"

sensorData = dataUtility.sensorData()
sensorData.loadJsonData(dataPath)

#data.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file


#FFT Analysis
def fftAnalysis(data:pd.DataFrame, print):
    #####################################################################################
    outlierDetector = dO.outlierDetector(data,'date')
    exteTempOutliers = outlierDetector.detectOutlierFFT('Ext_Tem',8,20,printFigure=print)
    exteUmiOutliers = outlierDetector.detectOutlierFFT('Ext_Umi',1,10,printFigure=print)
    exteIntUmiOutliers = outlierDetector.detectOutlierFFT('Int_Umi',printFigure=print)
    exteIntPuOutliers = outlierDetector.detectOutlierFFT('Int_Pu1',printFigure=print)
    exteExt_VviOutliers = outlierDetector.detectOutlierFFT('Ext_Vvi',printFigure=print)
    #####################################################################################


#Peculiarity Analysis
def peculiarityAnalysis(data:pd.DataFrame, print):
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


fftAnalysis(sensorData.dataFrame, True)
peculiarityAnalysis(sensorData.dataFrame, True)

     
#data['dataTimeStamp'] = (data['dataTimeStamp'] - datetime(1970,1,1)).dt.total_seconds()
##data.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Ext_Tem']]
#results_ExtTemp = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_ExtTemp,'Ext_Tem')

#np.datetime64(results_ExtTemp['anoms'].index[0]).tolist()

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Int_Tem']]
#results_Int_Tem = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Int_Tem,'Int_Tem')

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Ext_Umi']]
#results_Ext_Umi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Ext_Umi,'Ext_Umi')

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Int_Umi']]
#results_Int_Umi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Int_Umi,'Int_Umi')

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Int_Pu1']]
#results_Int_Pu1 = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Int_Pu1,'Int_Pu1')

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Int_Pu2']]
#results_Int_Pu2 = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Int_Pu2,'Int_Pu2')

##region ExtTemp outliers
#twoColumnsFrame = data[['dataTimeStamp','Ext_Vvi']]
#results_Ext_Vvi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
##plotOutliers(twoColumnsFrame,results_Ext_Vvi,'Ext_Vvi')

#totalOutliers = [results_ExtTemp, results_Int_Tem, results_Ext_Umi, results_Int_Umi, results_Int_Pu2, results_Ext_Vvi]

        



#Mahalanobis Analysis
#Principal Component Analysis

