# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:44:31 2020

@author: wen
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ch = pd.read_csv(r'./ch.csv')
de = pd.read_csv(r'./de.csv')
fr = pd.read_csv(r'./fr.csv')
nl = pd.read_csv(r'./nl.csv')

ch = ch.rename(columns={"Order creation date": "ds", "Quantity": "y"})
de = de.rename(columns={"Order creation date": "ds", "Quantity": "y"})
fr = fr.rename(columns={"Order creation date": "ds", "Quantity": "y"})
nl = nl.rename(columns={"Order creation date": "ds", "Quantity": "y"})

#%%
ch_train = nl[:63]
ch_test = nl[63:]

model = Prophet()
model.fit(ch_train);
#future = model.make_future_dataframe(periods=1, freq = 'w')
future = model.make_future_dataframe(periods=14)
future.tail()
#forecast = model.predict(future)
forecast_train = model.predict(future[:63])
forecast_test = model.predict(future[63:70])
forecast_val = model.predict(future[70:77])
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#model.plot(forecast)

#%%
prediction_train = forecast_train['yhat'].values.tolist()
prediction_test = forecast_test['yhat'].values.tolist()
prediction_val = forecast_val['yhat'].values.tolist()
gt_train = ch_train['y'].values.tolist()
gt_test = ch_test['y'].values.tolist()
mape_train = mean_absolute_percentage_error(gt_train, prediction_train)
mape_test = mean_absolute_percentage_error(gt_test, prediction_test)
print("mape(train): {}".format(mape_train))
print("mape(test): {}".format(mape_test))

# %%
# plot training result
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 63, 63)
ax.plot(x,gt_train,label="Ground Truth")
ax.plot(x, prediction_train,label="Prediction");
fig.suptitle('Ground Truth vs Prediction\n(Training)', fontsize=16)
plt.legend()
plt.draw()

fig = plt.figure()
ax = plt.axes()
x = np.linspace(63, 70, 7)
ax.plot(x,gt_test,label="Ground Truth")
ax.plot(x, prediction_test,label="Prediction");
fig.suptitle('Ground Truth vs Prediction\n(Test)', fontsize=16)
#fig.suptitle('Prediction for day 70-77', fontsize=16)
plt.legend()
plt.draw()

fig = plt.figure()
ax = plt.axes()
x = np.linspace(70, 77, 7)
#ax.plot(x,gt_test,label="Ground Truth")
ax.plot(x, prediction_val,label="Prediction");
#fig.suptitle('Ground Truth vs Prediction\n(Test)', fontsize=16)
fig.suptitle('Prediction for day 70-77 (Netherland)', fontsize=16)
plt.legend()
plt.draw()