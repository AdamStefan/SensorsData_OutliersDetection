import json
import os
import sys
import pandas as pd
import detectOutliers as dO
import matplotlib.pyplot as plt
import numpy as np

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

dataPath = "C:\\Users\\Stefan.Adam\\Desktop\\Proiect Cercetare\\data2"

data = loadJsonData(dataPath)

##################################################################################################################################
columnsNames = ['Ext_Tem','Ext_Tem_units','Ext_Umi','Ext_Umi_units','Ext_Vvi','Ext_Vvi_units','Int_Pu1','Int_Pu1_units','Int_Pu2',
                'Int_Pu2_units','Int_Tem','Int_Tem_histe','Int_Tem_ref','Int_Tem_units','Int_Umi','Int_Umi_histe','Int_Umi_ref',
                'Int_Umi_units','dataId','dataTimeStamp','id','recordStamp']
##################################################################################################################################

data['dataTimeStamp'] = pd.to_datetime(data['dataTimeStamp'],format='%Y-%m-%d_%H:%M:%S')
data.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file


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

######################################################################
exteTempOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Tem',8,20)
exteUmiOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Umi',1,10)
exteIntUmiOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Int_Umi')
exteIntPuOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Int_Pu1')
exteExt_VviOutliers = dO.detectOutlierFFT(data,'dataTimeStamp','Ext_Vvi')
#######################################################################


#Mahalanobis Analysis
#Principal Component Analysis

