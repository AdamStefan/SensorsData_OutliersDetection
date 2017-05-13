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

class sensorData(object):
    def __init__(self):        
        self.__data = None
        ##################################################################################################################################
        self.columnsNames = ['Ext_Tem','Ext_Tem_units','Ext_Umi','Ext_Umi_units','Ext_Vvi','Ext_Vvi_units','Int_Pu1','Int_Pu1_units','Int_Pu2',
                        'Int_Pu2_units','Int_Tem','Int_Tem_histe','Int_Tem_ref','Int_Tem_units','Int_Umi','Int_Umi_histe','Int_Umi_ref',
                        'Int_Umi_units','dataId','dataTimeStamp','id','recordStamp']
        ##################################################################################################################################
    
    def loadJsonData(self, jsonDataFolder):
        sensorData = [] 
        for root, dirnames, filenames in os.walk(jsonDataFolder):
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
        self.__data = pd.DataFrame(sensorData)
        self.__data['dataTimeStamp'] = pd.to_datetime(self.__data['dataTimeStamp'],format='%Y-%m-%d_%H:%M:%S')
        self.__data['date'] = self.__data['dataTimeStamp']
        self.__data['dataTimeStamp'] = (self.__data['dataTimeStamp'] - datetime(1970,1,1)).dt.total_seconds()
                     
        return self.__data

    @property
    def dataFrame(self):
        return self.__data

    def plotColumn(self, x, y):
        plt.xlabel(x)
        plt.ylabel(y)
        xValues = self.dataFrame[x].values
        yValues = self.dataFrame[y].values
        plt.plot(xValues, yValues)    

        title = y + ' versus ' + x
        plt.title(title)
        plt.grid(True)
        plt.savefig(title+".png")
        plt.show()

    def plotColumns(self, x, columns):
        f, axarr = plt.subplots(len(columns), sharex=True)
        plt.xlabel(x)
        xValues = self.dataFrame[x].values
        for  i in range(len(columns)):
            yValues = dataFrame[columns[i]].values
            title = columns[i]
            axarr[i].plot(xValues, yValues)
            axarr[i].set_title(title)        
        figTitle = ' '.join(columns)    
        plt.grid(True)
        plt.savefig(figTitle+".png")
        plt.show()


    def plotIndividualColumn(self):
        #################################################
        self.plotColumn('dataTimeStamp','Ext_Tem')
        self.plotColumn('dataTimeStamp','Ext_Umi')
        self.plotColumn('dataTimeStamp','Ext_Vvi')
        self.plotColumn('dataTimeStamp','Int_Pu1')
        self.plotColumn('dataTimeStamp','Int_Pu2')
        self.plotColumn('dataTimeStamp','Int_Tem')
        self.plotColumn('dataTimeStamp','Int_Umi')
        #################################################

    def plotColumns(self):
        #################################################
        self.plotColumns('dataTimeStamp',['Ext_Tem','Int_Tem']) ## strong correlated
        self.plotColumns('dataTimeStamp',['Ext_Umi','Int_Umi']) 
        self.plotColumns('dataTimeStamp',['Int_Pu1','Int_Pu2']) ## strong correlated   
        self.plotColumns('dataTimeStamp',['Ext_Tem','Int_Tem','Ext_Umi','Int_Umi','Ext_Vvi','Int_Pu1','Int_Pu2'])

def get_median_filtered(signal, threshold = 3):    
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal        