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
import arima_analysis as ar

data_path = "..\\data2"

#sensorData = dataUtility.sensorData()
#sensorData.loadJsonData(dataPath)
#sensorData.dataFrame.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file
#frame_intervals = sensorData.computeFramesIntervals()



#interval_index = 2

#number_of_steps = 20000
#number_of_points = 2000
#columns = ['Ext_Tem','Ext_Umi','Ext_Vvi','Int_Pu1','Int_Pu2','Int_Tem','Int_Umi']
#columns = ['Ext_Tem','Ext_Vvi','Int_Tem']

#model_fit = ar.fit_arima_and_predict(frame_intervals[interval_index],'date',columns,10,1,1)
#print(model_fit.sumary)



#train = False
#if train:
#    estimator =  nn_ts.train_and_predict(frame_intervals[interval_index], columns, number_of_points, number_of_steps)
#observed, prediction =  nn_ts.predict(frame_intervals[interval_index], columns)

#outlierDetector = dO.outlierDetector(frame_intervals[interval_index],'date')        
#outlierDetector.detect_outlier_from_prediction(frame_intervals[interval_index]['date'].values, observed, prediction, len(frame_intervals[interval_index])-number_of_points, columns)


#analysis = da.dataAnalysis(sensorData)


#analysis.timeAnalysis(False)
##analysis.fftWindowAnalysis()
##analysis.fftAnalysis(True,False)
#analysis.fftAnalysis_interval(True,False,3,columns)

##analysis.peculiarityAnalysis(sensorData.dataFrame, True)
     

##Mahalanobis Analysis
##Principal Component Analysis


def predict_values(method, frame_interval, columns, parameters):
    if method == 'ARIMA':
        observations, predictions, models_fit = ar.fit_arima_and_predict(frame_interval, 'date', columns, parameters)
        return observations, predictions
    if method == 'LSTM':
        force_train = parameters['force_train']                
        if force_train:                 
            estimator =  nn_ts.train_and_predict(frame_interval, columns, parameters)
        observations, predictions =  nn_ts.predict(frame_interval, columns)
        return observations, predictions

def detect_outlier_from_prediction(frame_interval, observations, predictions, columns, number_of_points, title):
    outlier_detector = dO.outlierDetector(frame_interval,'date')        
    outlier_detector.detect_outlier_from_prediction(frame_interval['date'].values, observations, predictions, len(frame_interval)-number_of_points, columns, title = title)

def predict_and_detect_outliers(method:str, sensor_data:dataUtility.sensorData, columns:list, parameters):
    frame_intervals = sensor_data.computeFramesIntervals()
    selected_index=-1
    max_frame_size = 0
    for i in range(len(frame_intervals)):
        if len(frame_intervals[i])> max_frame_size:
            selected_index = i
            max_frame_size = len(frame_intervals[i])
    frame_interval = frame_intervals[selected_index]
    
    observations, predictions = predict_values(method, frame_interval, columns, parameters)
    number_of_points = parameters["number_of_points_used_for_training"]
    detect_outlier_from_prediction(frame_interval, observations, predictions, columns, number_of_points, method)
       


if __name__ == "__main__":
    ### Load The data ###    
    sensor_data = dataUtility.sensorData()
    sensor_data.loadJsonData(data_path)

    ##Save DATA to csv for later view
    sensor_data.dataFrame.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file


    ### Which columns in which to detect outliers
    columns = ['Ext_Tem','Ext_Vvi','Int_Tem']


    arima_detect_parameters = {
                                "definition":(15,1,1), #p,d,q parameters for arima models
                                "number_of_points_used_for_training":2000
                               }
    
    nn_parameters = {
                      "force_train":False,
                      "number_of_steps":20000,
                      "number_of_points_used_for_training":2000
                     }

    ### Detect Outliers From Predictions
    predict_and_detect_outliers("ARIMA", sensor_data, columns, arima_detect_parameters)

    predict_and_detect_outliers("LSTM", sensor_data, columns, nn_parameters)



    ### Compute data analysis

    analysis = da.dataAnalysis(sensor_data)

    analysis.timeAnalysis(False)
    analysis.fftWindowAnalysis()
    analysis.fftAnalysis(True,False)
    analysis.fftAnalysis_interval(True,False,3,columns)
    analysis.peculiarityAnalysis(sensor_data.dataFrame, True)

    












