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

#################################################################################################################################
def loadJsonData(dataFolder):
    sensorData = [] 
    for root, dirnames, filenames in os.walk(dataFolder):
        for filename in filenames:
            fileNameWIthPath = os.path.join(root,filename)
            with open(fileNameWIthPath) as data_file:    
                data = json.load(data_file)
                item = {}
                item['id'] = data['id']
                item ['dataId'] = data['record']['sdata'][0]['ids']
                item ['recordStamp'] = data['record']['sdata'][0]['timestamp']
                item ['dataTimeStamp'] = data['timestamp'][:19]
                for sensor in data['record']['sdata'][0]['sensors']:
                    type = sensor['stype']
                    if type == 'Int_Tem' or type == 'Int_Umi':
                        item[type+'_ref'] = sensor['refer']
                        item[type+'_histe'] = sensor['histe']

                    item[type] = sensor['value']
                    item[type+'_units'] = sensor['units']
                sensorData.append(item)

    #sensorData = pd.Series(sensorData) #if we want to transform data to a series
    sensorData = pd.DataFrame(sensorData)     
    return sensorData

dataPath = "..\\data2"

data = loadJsonData(dataPath)

##################################################################################################################################
columnsNames = ['Ext_Tem','Ext_Tem_units','Ext_Umi','Ext_Umi_units','Ext_Vvi','Ext_Vvi_units','Int_Pu1','Int_Pu1_units','Int_Pu2',
                'Int_Pu2_units','Int_Tem','Int_Tem_histe','Int_Tem_ref','Int_Tem_units','Int_Umi','Int_Umi_histe','Int_Umi_ref',
                'Int_Umi_units','dataId','dataTimeStamp','id','recordStamp']
##################################################################################################################################

data['dataTimeStamp'] = pd.to_datetime(data['dataTimeStamp'],format='%Y-%m-%d_%H:%M:%S')
data['date'] = data['dataTimeStamp']
#data.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file


#twoColumnsFrame = data[['dataTimeStamp','Ext_Tem']]
#resultsExtTemp = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')


def plotIndividualColumn(data:pd.DataFrame):
    #################################################
    dO.plotColumn(data,'dataTimeStamp','Ext_Tem')
    dO.plotColumn(data,'dataTimeStamp','Ext_Umi')
    dO.plotColumn(data,'dataTimeStamp','Ext_Vvi')
    dO.plotColumn(data,'dataTimeStamp','Int_Pu1')
    dO.plotColumn(data,'dataTimeStamp','Int_Pu2')
    dO.plotColumn(data,'dataTimeStamp','Int_Tem')
    dO.plotColumn(data,'dataTimeStamp','Int_Umi')
    #################################################

def plotColumns(data:pd.DataFrame):
    #################################################
    dO.plotColumns(data,'dataTimeStamp',['Ext_Tem','Int_Tem']) ## strong correlated
    dO.plotColumns(data,'dataTimeStamp',['Ext_Umi','Int_Umi']) 
    dO.plotColumns(data,'dataTimeStamp',['Int_Pu1','Int_Pu2']) ## strong correlated   
    dO.plotColumns(data,'dataTimeStamp',['Ext_Tem','Int_Tem','Ext_Umi','Int_Umi','Ext_Vvi','Int_Pu1','Int_Pu2'])
    #################################################



#FFT Analysis
def fftAnalysis(data:pd.DataFrame):
    ######################################################################
    exteTempOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Tem',8,20)
    exteUmiOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Umi',1,10)
    exteIntUmiOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Int_Umi')
    exteIntPuOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Int_Pu1')
    exteExt_VviOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Vvi')
    #######################################################################


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


     
data['dataTimeStamp'] = (data['dataTimeStamp'] - datetime(1970,1,1)).dt.total_seconds()
#data.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Ext_Tem']]
results_ExtTemp = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_ExtTemp,'Ext_Tem')

np.datetime64(results_ExtTemp['anoms'].index[0]).tolist()

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Int_Tem']]
results_Int_Tem = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Int_Tem,'Int_Tem')

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Ext_Umi']]
results_Ext_Umi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Ext_Umi,'Ext_Umi')

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Int_Umi']]
results_Int_Umi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Int_Umi,'Int_Umi')

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Int_Pu1']]
results_Int_Pu1 = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Int_Pu1,'Int_Pu1')

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Int_Pu2']]
results_Int_Pu2 = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Int_Pu2,'Int_Pu2')

#region ExtTemp outliers
twoColumnsFrame = data[['dataTimeStamp','Ext_Vvi']]
results_Ext_Vvi = detect_ts(twoColumnsFrame, max_anoms=0.05, alpha=0.001, direction='both')
#plotOutliers(twoColumnsFrame,results_Ext_Vvi,'Ext_Vvi')

totalOutliers = [results_ExtTemp, results_Int_Tem, results_Ext_Umi, results_Int_Umi, results_Int_Pu2, results_Ext_Vvi]
dates = []

data['isOutlier'] = 0

for outliers in totalOutliers:
    for i in range(len(outliers)):
        dates.append(outliers['anoms'].index[i])
        data['isOutlier'][data['date']==outliers['anoms'].index[i]] = 1


data.to_csv('sensorDataWithOutliers.csv', sep=',', encoding='utf-8') # save the data to a csv file
        







#Mahalanobis Analysis
#Principal Component Analysis

