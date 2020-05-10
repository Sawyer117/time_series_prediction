# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:06:54 2020

@author: Wen
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error
#date_now = datetime.now().date()
#start_date = date_now + relativedelta(days=+1)
#date_time_str = '2018-06-29 17:08:00'
#date_time_obj = dt.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
#%%
df = pd.read_excel('Project1.xlsx')    #sheetname='Sheet1'

print("Column headings:")
print(df.columns)

## Get one-hot encoding for country names
countryList = df["Shipping country"].unique()
countryFeature = pd.get_dummies(countryList)

#def get_label_from_creation_time(df):
#    creation_time = 
#def plot(x):
#   plt.plot(x)
#   plt.show()
    
#%%
# =============================================================================
# # obsele , too slow and cumbersome
# def prepare_data_by_date(df):
#     #column_names = ['Day','Country','Quantity']
#     #data_frame = pd.DataFrame(columns=column_names)
#     start_label =  str(df['Order creation date'].min())
#     end_label = str(df['Order creation date'].max())
#     day_now = 1
#     
#     column_names = ['ds','y']
#     ch_frame = pd.DataFrame(columns=column_names)
#     de_frame = pd.DataFrame(columns=column_names)
#     fr_frame = pd.DataFrame(columns=column_names)
#     nl_frame = pd.DataFrame(columns=column_names)
#     while True:
#         CH_quantity = df.loc[(df['Shipping country'] == "CH")].loc[df.loc[(df['Shipping country'] == "CH")].index.intersection(df.loc[df['Order creation date']==start_label].index)]["Quantity"].sum()
#         DE_quantity = df.loc[(df['Shipping country'] == "DE")].loc[df.loc[(df['Shipping country'] == "DE")].index.intersection(df.loc[df['Order creation date']==start_label].index)]["Quantity"].sum()
#         FR_quantity = df.loc[(df['Shipping country'] == "FR")].loc[df.loc[(df['Shipping country'] == "FR")].index.intersection(df.loc[df['Order creation date']==start_label].index)]["Quantity"].sum()
#         NL_quantity = df.loc[(df['Shipping country'] == "NL")].loc[df.loc[(df['Shipping country'] == "NL")].index.intersection(df.loc[df['Order creation date']==start_label].index)]["Quantity"].sum()
#         #data_frame = data_frame.append({'Day':day_now,'Country':"CH",'Quantity':CH_quantity},ignore_index=True)
#         #data_frame = data_frame.append({'Day':day_now,'Country':"DE",'Quantity':DE_quantity},ignore_index=True)
#         #data_frame = data_frame.append({'Day':day_now,'Country':"FR",'Quantity':FR_quantity},ignore_index=True)
#         #data_frame = data_frame.append({'Day':day_now,'Country':"NL",'Quantity':NL_quantity},ignore_index=True)
#         
#         if datetime.strptime(start_label, '%Y-%m-%d %H:%M:%S')==datetime.strptime(end_label, '%Y-%m-%d %H:%M:%S'):
#             break
#         start_label = str(datetime.strptime(start_label, '%Y-%m-%d %H:%M:%S')+ relativedelta(days=+1))
#         day_now += 1
#     return data_frame
# =============================================================================
# =============================================================================

#%%
def prepare_data_by_date2(df):
     aggregation_functions = {'Quantity': 'sum', 'Order creation date': 'first'}
     ch = df.loc[(df['Shipping country'] == "CH")]
     de = df.loc[(df['Shipping country'] == "DE")]
     fr = df.loc[(df['Shipping country'] == "FR")]
     nl = df.loc[(df['Shipping country'] == "NL")]
     ch = ch.groupby(ch['Order creation date']).aggregate(aggregation_functions)
     de = de.groupby(de['Order creation date']).aggregate(aggregation_functions)
     fr = fr.groupby(fr['Order creation date']).aggregate(aggregation_functions)
     nl = nl.groupby(nl['Order creation date']).aggregate(aggregation_functions)
     columns_titles = ['Order creation date','Quantity']
     ch=ch.reindex(columns=columns_titles)
     de=de.reindex(columns=columns_titles)
     fr=fr.reindex(columns=columns_titles)
     nl=nl.reindex(columns=columns_titles)
     ch.to_csv(r'./ch.csv', index = False)
     de.to_csv(r'./de.csv', index = False)
     fr.to_csv(r'./fr.csv', index = False)
     nl.to_csv(r'./nl.csv', index = False)

prepare_data_by_date2(df)
#%%
#total_data = prepare_data_by_date(df)
#total_data.to_csv(r'./total_data.csv', index = False)
#%%
# =============================================================================
# plt.plot(list(total_data.loc[(total_data['Country'] == "CH")]["Quantity"]),label="CN")
# plt.plot(list(total_data.loc[(total_data['Country'] == "DE")]["Quantity"]),label="DE")
# plt.plot(list(total_data.loc[(total_data['Country'] == "FR")]["Quantity"]),label="FR")
# plt.plot(list(total_data.loc[(total_data['Country'] == "NL")]["Quantity"]),label="NL")
# plt.legend()
# plt.show()
# 
# 
# 
# train_data = total_data[:252]
# test_data = total_data[252:]
# 
# 
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# 
# # Train the model on training data
# train_features = np.reshape(np.linspace(1,63, num=63),(-1,1))
# #train_labels = list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"])
# train_labels = np.reshape(list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)
# #train_labels = list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"])
# rf.fit(train_features, train_labels);
# l_reg = LinearRegression().fit(train_features, train_labels)
# svm_reg=SVR(kernel='rbf',epsilon=1.0)
# svm_reg.fit(train_features, train_labels)
# #prediction=svm_reg.predict(train_features) #print(svm_reg.score(train_features, train_labels))
# prediction = rf.predict(train_features)
# #prediction = l_reg.predict(train_features)
# #%%
# #compute accuracy of training 
# #accuracy = accuracy_score(train_labels,np.reshape(prediction,(-1,1)))
# mse = mean_squared_error(train_labels.astype(np.float32), prediction)
# mse2 = np.square(np.subtract(train_labels,prediction)).mean() 
# # %%
# # plot training result
# fig = plt.figure()
# ax = plt.axes()
# #x = np.linspace(0, 63, 63)
# ax.plot(train_features, train_labels,label="Ground Truth");
# ax.plot(train_features, prediction,label="Prediction");
# fig.suptitle('Ground Truth vs Prediction\n(Training)', fontsize=16)
# plt.legend()
# plt.draw()
# 
# # %%
# test_features = np.linspace(63,70, num=7)
# prediction_test = rf.predict(np.reshape(test_features,(-1,1)))
# #prediction_test = l_reg.predict(np.reshape(test_features,(-1,1)))
# test_labels = np.reshape(list(test_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)
# fig = plt.figure()
# ax = plt.axes()
# #x = np.linspace(0, 63, 63)
# ax.plot(test_features, test_labels,label="Ground Truth");
# ax.plot(test_features, prediction_test,label="Prediction");
# fig.suptitle('Ground Truth vs Prediction\n(Test)', fontsize=16)
# plt.legend()
# plt.draw()
# =============================================================================
