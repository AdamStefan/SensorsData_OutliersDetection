import numpy as np
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pyculiarity import detect_ts
from anomaly_detection import anomaly_detect_ts as detts
from datetime import datetime

COLOR_PALETTE = [    
               "#348ABD",
               "#A60628",
               "#7A68A6",
               "#467821",
               "#CF4457",
               "#188487",
               "#E24A33"
              ]


class outlierDetector(object):

    def __init__(self, data:pd.DataFrame, xColumn):
        self.data = data
        self.xColumn = xColumn
   
    def __detect_outlier_position_by_fft(self, signal, threshold_freq=.1, frequency_amplitude=.01):
        fft_of_signal = np.fft.fft(signal)
        outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
        if np.any(np.abs(fft_of_signal[int(threshold_freq):]) > frequency_amplitude):
            index_of_outlier = np.where(signal == outlier)
            return index_of_outlier[0]
        else:
            return None
        
    def display_data_and_outliers(time, frame, columns, outliers, start_step, title):
        plt.figure(figsize=(15, len(columns)))

        if start_step > 0:
            plt.axvline(time[start_step], linestyle="dotted", linewidth=4, color='g')

        handles = [] 
        for feature_index in range(len(columns)):
           handle_obs = plt.plot(time,frame[columns[feature_index]].values,color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-observation")           
           handles.append(handle_obs[0])
                   
        plt.legend(handles=handles, loc="upper left")
        
        for index in range(len(outliers)):
            time_value = outliers['Date'][index]
            feature = outliers['Feature'][index]
            print_outlier = True
            if (start_step>=0 and time[start_step]>np.datetime64(time_value)) or time[-1]<np.datetime64(time_value):
                print_outlier = False
            if print_outlier:
                itemindex = np.where(time==np.datetime64(time_value))
                plt.plot(time_value,frame[feature].values[itemindex[0][0]],'ro')        
        plt.savefig(title+".png")
        plt.show()
        

    def detect_outlier_fft(self, yColumn, threshold_freq=8, frequency_amplitude = 20, windowSize = 10, printFigure = True):        
        figTitle = yColumn +" - Outliers using FFT for threshold_freq="+str(threshold_freq)+" and frequency_amplitude="+ str(frequency_amplitude)
        yValues = self.data[yColumn].values
        xValues = self.data[self.xColumn].values   

        outlier_positions = []
        for ii in range(int(windowSize/2), yValues.size, int(windowSize/2)):
            outlier_position = self.__detect_outlier_position_by_fft(yValues[ii-int(windowSize/2):ii+int(windowSize/2)],threshold_freq,frequency_amplitude)
            if outlier_position is not None:
                outlier_positions.append(ii + outlier_position[0] - int(windowSize/2))
        outlier_positions = list(set(outlier_positions))

        if (printFigure):
            plt.figure(figsize=(12, 6));
            plt.xlabel(self.xColumn)
            plt.ylabel(yColumn)    
    
            plt.plot(xValues, yValues, c=COLOR_PALETTE[0], label='Original Signal');
            if len(outlier_positions) > 0:
                plt.plot(xValues[outlier_positions], yValues[np.asanyarray(outlier_positions)], 'ro')
    
            plt.title(figTitle)
    
            plt.legend();
            plt.savefig(figTitle+".png")
            plt.show()
        return outlier_positions

    def detect_outlier_from_prediction(self, time, observed_data, predicted_data, last_k_steps, columns, print_outliers = True, print_only_validation = False, title='outlier_detection'):

        mean_values = np.mean(observed_data[0:len(observed_data)- last_k_steps],axis=0)
        std_values = np.std(observed_data[0:len(observed_data)- last_k_steps],axis=0)
        epsilon = 1e-8
        observed_data_normalized = (observed_data-mean_values)/(std_values + epsilon)
        predicted_data_normalized = (predicted_data -mean_values)/(std_values + epsilon)
        mean_squared_error = 0.0

        
        plt.figure(figsize=(15, len(mean_values)))
        plt.axvline(time[len(observed_data)- last_k_steps], linestyle="dotted", linewidth=4, color='g')
        
        handles = [] 
        for feature_index in range(len(mean_values)):
            if print_only_validation:
                handle_obs = plt.plot(time[-last_k_steps:],observed_data[-last_k_steps:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-observation")
                handle_pred = plt.plot(time[-last_k_steps:],predicted_data[-last_k_steps:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-prediction", linestyle='dotted')
            else:
                handle_obs = plt.plot(time,observed_data[:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-observation")
                handle_pred = plt.plot(time,predicted_data[:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-prediction", linestyle='dotted')
            
            handles.append(handle_obs[0])
            handles.append(handle_pred[0])
        
        plt.legend(handles=handles, loc="upper left")               

        outlier_positions=[]        
        acumulator = 0
        for i in range(last_k_steps):
            index= len(observed_data)- last_k_steps + i
            error = np.power(observed_data[index]  - predicted_data[index],2)
            mean_squared_error = ((i)/(i+1.0))* mean_squared_error + error/(i+1.0)


            observed_value_at_index = observed_data_normalized[index]
            predicted_data_at_index = predicted_data_normalized[index]
            for feature_index in range(len(mean_values)):
                error = abs((predicted_data_at_index[feature_index]-observed_value_at_index[feature_index]))
                if error > 3 :
                    acumulator =  acumulator + error                   
                else:
                    acumulator = acumulator/2 
                if acumulator > 5:
                    #Date
                    outlier_item = {}
                    outlier_item["Date"] = time[index]
                    outlier_item["Feature"] = columns[feature_index]
                    outlier_positions.append(outlier_item)
                    if print_outliers:
                        plt.plot(time[index],observed_data[index][feature_index],'ro')
                    print("Outlier detected on index",index, "for feature", columns[feature_index])
                    print(error,acumulator)
                    print(observed_data[index][feature_index],predicted_data[index][feature_index])
                    print(observed_value_at_index[feature_index],predicted_data_at_index[feature_index])
                    print("Mean value",mean_values[feature_index],"Variance",std_values[feature_index])
                    print("----------------------------------------------------------------------")
        plt.title(title)        
        plt.savefig(title+".png")
        plt.show()        
        if len(outlier_positions)>0:
            return pd.DataFrame(outlier_positions), mean_squared_error
        return None, mean_squared_error

                
                                
    def detect_outlier_peculiarity(self, frame, yColumn, max_anoms = 0.05, alpha = 0.001, direction='both', printFigure=True):        

        def plotOutliers(data, results, columnName):
            # format the data nicely
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', drop=True)

            # make a nice plot
            f, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(data['timestamp'], data[columnName], 'b')
            ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
            ax[0].set_title('Detected Anomalies')
            ax[1].set_xlabel('Time Stamp')
            ax[0].set_ylabel(columnName)
            ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
            ax[1].set_ylabel('Anomaly Magnitude')
            figTitle = columnName + " - Outliers using TwitterDetector"
            plt.savefig(figTitle+".png")
            plt.show()

        frame = frame.copy()
        
        frame['timestamp'] = (frame['date'] - datetime(1970,1,1)).dt.total_seconds()

        twoColumnsFrame = frame[['timestamp', yColumn]]

        #s = twoColumnsFrame.set_index('date')[yColumn]
        #results = detts.anomaly_detect_ts(s, max_anoms=0.05, alpha=0.001, direction='both')        
        try:
            results = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')        
        except Exception as e:
            return []

        if (printFigure):
            plotOutliers(twoColumnsFrame, results, yColumn)                

        return results
            












             


    