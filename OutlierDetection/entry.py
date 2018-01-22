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
import dataAnalysis as da
import neural_net_time_series as nn_ts

dataPath = "..\\data2"

sensorData = dataUtility.sensorData()
sensorData.loadJsonData(dataPath)
sensorData.dataFrame.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file
frame_intervals = sensorData.computeFramesIntervals()
#nn_ts.train_and_predict(frame_intervals[3],['Ext_Tem','Ext_Umi','Ext_Vvi','Int_Pu1','Int_Pu2','Int_Tem','Int_Umi'])

##'Ext_Umi','Int_Pu1' ,'Int_Pu2'problema
#nn_ts.train_and_predict(frame_intervals[3],['Ext_Tem','Ext_Vvi','Int_Tem'])
number_of_steps = 20000
number_of_points = 2000
columns = ['Ext_Tem','Ext_Umi','Ext_Vvi','Int_Pu1','Int_Pu2','Int_Tem','Int_Umi']
columns = ['Ext_Tem','Ext_Vvi','Int_Tem']
train = False
if train:
    estimator =  nn_ts.train_and_predict(frame_intervals[3], columns, number_of_points, number_of_steps)
observed, prediction =  nn_ts.predict(frame_intervals[3], columns)

outlierDetector = dO.outlierDetector(frame_intervals[3],'date')        
outlierDetector.detect_outlier_from_prediction(frame_intervals[3]['date'].values, observed,prediction,len(frame_intervals[3])-number_of_points,columns)


#frameIntervals = sensorData.computeFramesIntervals()
analysis = da.dataAnalysis(sensorData)


analysis.timeAnalysis(False)
#analysis.fftWindowAnalysis()
#analysis.fftAnalysis(True,False)
analysis.fftAnalysis_interval(True,False,3,columns)

#analysis.peculiarityAnalysis(sensorData.dataFrame, True)
     

#Mahalanobis Analysis
#Principal Component Analysis

