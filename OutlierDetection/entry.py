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

dataPath = "..\\data2"

sensorData = dataUtility.sensorData()
sensorData.loadJsonData(dataPath)
sensorData.dataFrame.to_csv('sensorData.csv', sep=',', encoding='utf-8') # save the data to a csv file
#frameIntervals = sensorData.computeFramesIntervals()
analysis = da.dataAnalysis(sensorData)


analysis.timeAnalysis(False)
#analysis.fftWindowAnalysis()
analysis.fftAnalysis(True,False)

#analysis.peculiarityAnalysis(sensorData.dataFrame, True)
     

#Mahalanobis Analysis
#Principal Component Analysis

