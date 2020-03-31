# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:50:55 2020

@author: GOYAL
"""

import pandas as pd
import numpy as np
import math


data = pd.read_csv('studentscores.csv')

data.shape
data.isnull().sum()

X = data.iloc[:, :1]
y = data.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# here rmse:  4.5092043283688055

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# 
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model = model.fit(X_train, y_train)
# 
# y_pred = model.predict(X_test)
# 
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# # here rmse: 20.109715964531997
# =============================================================================

import matplotlib.pyplot as plt
# visualizing training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')

# visualizing testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test), color='blue')
