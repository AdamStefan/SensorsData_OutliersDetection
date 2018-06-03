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
outliers_path="..\\outliers.txt"


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

def detect_outlier_from_prediction(frame_interval, observations, predictions, columns, number_of_points, print_outliers, print_only_validation, title):
    outlier_detector = dO.outlierDetector(frame_interval,'date')        
    return outlier_detector.detect_outlier_from_prediction(frame_interval['date'].values, observations, predictions, len(frame_interval)-number_of_points, columns, print_outliers = print_outliers, print_only_validation = print_only_validation , title = title)

def predict_and_detect_outliers(method:str, frame_interval:pd.DataFrame, columns:list, parameters):       
    observations, predictions = predict_values(method, frame_interval, columns, parameters)
    number_of_points = parameters["number_of_points_used_for_training"]
    if parameters["print_outliers"] == True:
        title = "Outliers using " + method
    else:
        title = "Prediction using " + method       
    outliers_predicted = detect_outlier_from_prediction(frame_interval, observations, predictions, columns, number_of_points, parameters["print_outliers"], parameters["print_only_validation"], title)
    return outliers_predicted
       
def compute_precision_and_recall(outliers, predicted_outliers, window_size, start_time, end_time):    
    outliers_status = np.zeros(len(outliers))
    predicted_outliers = predicted_outliers[predicted_outliers['Date']>=start_time]
    predicted_outliers = predicted_outliers[predicted_outliers['Date']<=end_time]
    
    predicted_outliers_status =  np.zeros(len(predicted_outliers))

    print("Computing precision")
    for predicted_index in range(len(predicted_outliers)):
        current_outlier_date    = predicted_outliers['Date'].values[predicted_index]
        current_outlier_feature = predicted_outliers['Feature'].values[predicted_index]        

        for index in range(len(outliers)):
            outlier_date    = outliers['Date'].values[index]
            outlier_feature = outliers['Feature'].values[index]
            if abs(outlier_date - current_outlier_date) <  window_size and current_outlier_feature == outlier_feature:
                outliers_status[index] = 1
                predicted_outliers_status[predicted_index] = 1                
                break        
            
    precision = np.sum(predicted_outliers_status)/len(predicted_outliers_status)
    recall = np.sum(outliers_status)/len(outliers_status)
    f_score = 2 * (precision * recall) /(precision + recall)

    return precision, recall, f_score
            

if __name__ == "__main__":
    ### Load The data ###    
    sensor_data = dataUtility.sensorData()
    sensor_data.loadJsonData(data_path)
    sensor_data.load_outliers(outliers_path)



    frame_intervals = sensor_data.computeFramesIntervals()
    selected_index=-1
    max_frame_size = 0
    for i in range(len(frame_intervals)):
        if len(frame_intervals[i])> max_frame_size:
            selected_index = i
            max_frame_size = len(frame_intervals[i])
    frame_interval = frame_intervals[selected_index]


    #####################################################

    
    #Interval chosen for bulding model is: 04/12/2015 16:15 - 29/12/2015 22:30 
    # Period for predicting outliers : 25/12/2015 16:15 - 29/12/2015 22:30
    # Human labeled:
    # Feature 
    #	- 25/12/2015 22:45 (Ext_Vvi) 
    #	- 28/12/2015 9:15  (Ext_Vvi)
    #	- 28/12/2015 19:15 (Ext_Tem)
    #	- 28/12/2015 20:15 (Ext_Tem)
    #	- 28/12/2015 21:15 (Ext_Tem)
    #	- 28/12/2015 22:00 (Ext_Tem)
    #	- 29/12/2015 3:00  (Ext_Vvi)

    #####################################################


    ##Save DATA to csv for later view
    sensor_data.dataFrame.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file


    ### Which columns in which to detect outliers
    columns = ['Ext_Tem','Ext_Vvi','Int_Tem']
    start_step = 2000
    print_only_validation = True


    arima_detect_parameters = {
                                "definition":(15,1,1), #p,d,q parameters for arima models
                                "number_of_points_used_for_training": start_step,
                                "print_outliers":True,
                                "print_only_validation":print_only_validation
                               }
    
    nn_parameters = {
                      "force_train":False,
                      "number_of_steps":20000,
                      "number_of_points_used_for_training": start_step,
                      "print_outliers":True,
                      "print_only_validation":print_only_validation
                     }

    ### Detect Outliers From Predictions

    dO.outlierDetector.display_data_and_outliers(frame_interval['date'].values[start_step:], frame_interval[start_step:], columns, sensor_data.outliers, start_step = 0, title = "Outliers")

    #region ARIMA

    #predicted_outliers_arima, mse = predict_and_detect_outliers("ARIMA", frame_interval, columns, arima_detect_parameters)
    #precision_arima, recall_arima, f_score_arima = compute_precision_and_recall(sensor_data.outliers, predicted_outliers_arima, window_size=sensor_data.step_size, start_time=frame_interval['date'].values[start_step], end_time=frame_interval['date'].values[-1])
    #print(precision_arima, recall_arima, f_score_arima)

    ##region neural network

    predicted_outliers_lstm, mse = predict_and_detect_outliers("LSTM", frame_interval, columns, nn_parameters)
    precision, recall, f_score = compute_precision_and_recall(sensor_data.outliers, predicted_outliers_lstm, window_size=sensor_data.step_size, start_time=frame_interval['date'].values[start_step], end_time=frame_interval['date'].values[-1])
    print(precision, recall, f_score)

    #endregion



    #### Compute data analysis

    analysis = da.data_analysis(sensor_data)
    analysis.time_analysis(False)


    # fft
    analysis.fft_window_Analysis()
    fft_outliers = analysis.fft_analysis_interval(True, False, frame_interval, columns)
    if print_only_validation == False:   
        dO.outlierDetector.display_data_and_outliers(frame_interval['date'].values, frame_interval, columns, fft_outliers, start_step, "Outliers using FFT")
    else:
        dO.outlierDetector.display_data_and_outliers(frame_interval['date'].values[start_step:], frame_interval[start_step:], columns, fft_outliers, 0, "Outliers using FFT")

    precision_fft, recall_fft, f_score_fft = compute_precision_and_recall(sensor_data.outliers, fft_outliers, window_size=sensor_data.step_size, start_time=frame_interval['date'].values[start_step], end_time=frame_interval['date'].values[-1])
    print(precision_fft, recall_fft, f_score_fft)


    # peculiarity
    peculiarity_outliers = analysis.peculiarity_analysis(frame_interval, columns, True)
    if print_only_validation == False:   
        dO.outlierDetector.display_data_and_outliers(frame_interval['date'].values, frame_interval, columns, peculiarity_outliers, start_step, "Outliers using Twitter algorithm")
    else:
        dO.outlierDetector.display_data_and_outliers(frame_interval['date'].values[start_step:], frame_interval[start_step:], columns, peculiarity_outliers, 0, "Outliers using Twitter algorithm")
    precision_peculiarity, recall_peculiarity, f_score_peculiarity = compute_precision_and_recall(sensor_data.outliers, peculiarity_outliers, window_size=sensor_data.step_size, start_time=frame_interval['date'].values[start_step], end_time=frame_interval['date'].values[-1])
    print(precision_peculiarity, recall_peculiarity, f_score_peculiarity)
