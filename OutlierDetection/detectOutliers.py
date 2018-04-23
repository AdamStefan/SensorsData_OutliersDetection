import numpy as np
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pyculiarity import detect_ts

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

    def detectOutlierFFT(self, yColumn, threshold_freq=8, frequency_amplitude = 20, windowSize = 10, printFigure = True):        
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

    def detect_outlier_from_prediction(self, time, observed_data, predicted_data, last_k_steps, columns, print_outliers = True, title='outlier_detection'):

        mean_values = np.mean(observed_data[0:len(observed_data)- last_k_steps],axis=0)
        std_values = np.std(observed_data[0:len(observed_data)- last_k_steps],axis=0)
        epsilon = 1e-8
        observed_data_normalized = (observed_data-mean_values)/(std_values + epsilon)
        predicted_data_normalized = (predicted_data -mean_values)/(std_values + epsilon)

        plt.title(title)
        plt.figure(figsize=(15, len(mean_values)))
        plt.axvline(time[len(observed_data)- last_k_steps], linestyle="dotted", linewidth=4, color='g')
        
        handles = [] 
        for feature_index in range(len(mean_values)):
           handle_obs = plt.plot(time,observed_data[:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-observation")
           handle_pred = plt.plot(time,predicted_data[:,feature_index],color=COLOR_PALETTE[feature_index],label=columns[feature_index]+"-prediction", linestyle='dotted')
           handles.append(handle_obs[0])
           handles.append(handle_pred[0])
        
        plt.legend(handles=handles, loc="upper left")               

        outlier_positions=[]
        acumulator = 0
        for i in range(last_k_steps):
            index= len(observed_data)- last_k_steps + i
            observed_value_at_index = observed_data_normalized[index]
            predicted_data_at_index = predicted_data_normalized[index]
            for feature_index in range(len(mean_values)):
                error = abs((predicted_data_at_index[feature_index]-observed_value_at_index[feature_index]))
                if error > 3 :
                    acumulator =  acumulator + error                   
                else:
                    acumulator = acumulator/2 
                if acumulator > 5:
                    outlier_positions.append(index)
                    if print_outliers:
                        plt.plot(time[index],observed_data[index][feature_index],'ro')
                    print("Outlier detected on index",index, "for feature", columns[feature_index])
                    print(error,acumulator)
                    print(observed_data[index][feature_index],predicted_data[index][feature_index])
                    print(observed_value_at_index[feature_index],predicted_data_at_index[feature_index])
                    print("Mean value",mean_values[feature_index],"Variance",std_values[feature_index])
                    print("----------------------------------------------------------------------")

        plt.show()
        return outlier_positions
                                
    def detectOutlierPeculiarity(self,yColumn,max_anoms = 0.05, alpha = 0.001, direction='both', printFigure=True):
        twoColumnsFrame = self.data[[self.xColumn, yColumn]]
        results = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')        

        if (printFigure):
            def plotOutliers(data,results, columnName):
                # format the data nicely
                data['dataTimeStamp'] = pd.to_datetime(data['dataTimeStamp'])
                data.set_index('dataTimeStamp', drop=True)

                # make a nice plot
                f, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(data['dataTimeStamp'], data[columnName], 'b')
                ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
                ax[0].set_title('Detected Anomalies')
                ax[1].set_xlabel('Time Stamp')
                ax[0].set_ylabel(columnName)
                ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
                ax[1].set_ylabel('Anomaly Magnitude')
                figTitle = columnName + " - Outliers using TwitterDetector"
                plt.savefig(figTitle+".png")
                plt.show()
            plotOutliers(twoColumnsFrame,results,yColumn)

        return results
            












             


    