import numpy as np
import pandas as pd
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


def fit_arima_and_predict(df, x_column_name, y_column_names, parameters):
    p, d, q = parameters["definition"]
    x_column = df[x_column_name].values
    models_fit = []
    predictions  = []
    observations = []
    for y_column_name in y_column_names:                
        y_column = df[y_column_name].values    
        model = ARIMA(y_column, order=(p,d,q))
        model_fit = model.fit(disp=0)

        y_predicted = model_fit.predict(d,len(y_column)-1)
        y_predicted = np.pad(y_predicted, (d,0), 'constant', constant_values=(0, 0))
        predictions.append(y_predicted)
        observations.append(y_column)
        models_fit.append(model_fit)

    predictions = np.asarray(predictions)
    observations = np.asarray(observations)
    return np.transpose(observations), np.transpose(predictions), models_fit
            





