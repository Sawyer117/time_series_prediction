# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:44:18 2020

@author: wen
"""
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error

# Release the DL
import keras
from keras import losses
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# read pre-processed data (this part of data came from process_data.py)
total_data = pd.read_csv(r'./total_data.csv')

#%%
plt.plot(list(total_data.loc[(total_data['Country'] == "CH")]["Quantity"]),label="CN")
plt.plot(list(total_data.loc[(total_data['Country'] == "DE")]["Quantity"]),label="DE")
plt.plot(list(total_data.loc[(total_data['Country'] == "FR")]["Quantity"]),label="FR")
plt.plot(list(total_data.loc[(total_data['Country'] == "NL")]["Quantity"]),label="NL")
plt.legend()
plt.show()



train_data = total_data[:252]
test_data = total_data[252:]
train_labels = np.reshape(list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)
test_labels = np.reshape(list(test_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)

train_labels_dnn = np.reshape(list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)/max(train_labels)
test_labels_dnn = np.reshape(list(test_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)/max(train_labels)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
train_features = np.reshape(np.linspace(1,63, num=63),(-1,1))
test_features = np.reshape(np.linspace(63,70, num=7),(-1,1))
#train_labels = list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"])
#train_labels = list(train_data.loc[(total_data['Country'] == "CH")]["Quantity"])
rf.fit(train_features, train_labels);
l_reg = LinearRegression().fit(train_features, train_labels)
svm_reg=SVR(kernel='rbf',epsilon=1.0)
svm_reg.fit(train_features, train_labels)
rr = Ridge(alpha=100) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr.fit(train_features, train_labels)
#prediction=svm_reg.predict(train_features) #print(svm_reg.score(train_features, train_labels))
prediction = rf.predict(train_features)
#prediction = l_reg.predict(train_features)
#prediction = rr.predict(train_features)


#%%
# Deep learning method
#  DNN-BASED Regression model
#  Model-building block
input_layer = keras.layers.Input(shape=train_features.shape[1:2])
a1=keras.layers.Dense(200, activation='sigmoid', use_bias=True,kernel_initializer='glorot_uniform')(input_layer)
a2=keras.layers.Dense(100, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')(a1)
#a2=keras.layers.Dropout(0.1, noise_shape=None, seed=None)(a2)
#a3=keras.layers.Dense(100, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(input_layer)
a4=keras.layers.Dense(50, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')(a2)
#a5=keras.layers.Dropout(0.1, noise_shape=None, seed=None)(a4)
output_layer = keras.layers.Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')(a4) 
model = Model(input_layer, output_layer)
#%%
adam=Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
sgd = keras.optimizers.SGD(learning_rate=0.001)
#adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)logcosh mean_absolute_percentage_error
model.compile(optimizer='adam', loss='mse', metrics=['mape','mse'])
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])#was dice_coef
# # #model.fit(x_train, y_train, epochs=1, batch_size=10,validation_split=0.2, shuffle=True, callbacks=[weight_saver])
savebest=ModelCheckpoint('BestDNN-max.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#history=model.fit(train_data.T,train_label_systolic,validation_data=(test_data.T,test_label_systolic),epochs=1000, batch_size=10,callbacks=[savebest])
history=model.fit(train_features,train_labels_dnn,validation_data=(test_features,test_labels_dnn),epochs=10000, batch_size=1,callbacks=[savebest])
prediction = model.predict(train_features)*max(train_labels)

#%%
# Load model and continue training
model = keras.models.load_model("DNN_iter10k.h")
adam=Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
sgd = keras.optimizers.SGD(learning_rate=0.001)
#adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005, amsgrad=False)logcosh mean_absolute_percentage_error
model.compile(optimizer='adam', loss='mse', metrics=['mape','mse'])
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])#was dice_coef
# # #model.fit(x_train, y_train, epochs=1, batch_size=10,validation_split=0.2, shuffle=True, callbacks=[weight_saver])
savebest=ModelCheckpoint('BestDNN-max.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#history=model.fit(train_data.T,train_label_systolic,validation_data=(test_data.T,test_label_systolic),epochs=1000, batch_size=10,callbacks=[savebest])
history=model.fit(train_features,train_labels_dnn,validation_data=(test_features,test_labels_dnn),epochs=10000, batch_size=1,callbacks=[savebest])
prediction = model.predict(train_features)*max(train_labels)
#%%
#compute accuracy of training 
#accuracy = accuracy_score(train_labels,np.reshape(prediction,(-1,1)))
mse = mean_squared_error(train_labels.astype(np.float32), prediction)
mse2 = np.square(np.subtract(train_labels,prediction)).mean() 
# %%
# plot training result
print("mape(train): {}".format(mean_absolute_percentage_error(train_labels, prediction)))
fig = plt.figure()
ax = plt.axes()
#x = np.linspace(0, 63, 63)
ax.plot(train_features, train_labels,label="Ground Truth");
ax.plot(train_features, prediction,label="Prediction");
fig.suptitle('Ground Truth vs Prediction\n(Training)', fontsize=16)
plt.legend()
plt.draw()

# %%
#prediction_test = rf.predict(test_features)
prediction_test = model.predict(test_features)*max(train_labels)
print("mape(test): {}".format(mean_absolute_percentage_error(test_labels, prediction_test)))
#prediction_test = l_reg.predict(np.reshape(test_features,(-1,1)))
#test_labels = np.reshape(list(test_data.loc[(total_data['Country'] == "CH")]["Quantity"]),(-1,1)).astype(np.float32)
fig = plt.figure()
ax = plt.axes()
#x = np.linspace(0, 63, 63)
ax.plot(test_features, test_labels,label="Ground Truth");
ax.plot(test_features, prediction_test,label="Prediction");
fig.suptitle('Ground Truth vs Prediction\n(Test)', fontsize=16)
plt.legend()
plt.draw()
