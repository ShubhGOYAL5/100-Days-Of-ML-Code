# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:46:05 2020

@author: GOYAL
"""

import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')

data.shape
# (400, 5)

data.isnull().sum()
# =============================================================================
# User ID            0
# Gender             0
# Age                0
# EstimatedSalary    0
# Purchased          0
# =============================================================================

data.dtypes
# =============================================================================
# User ID             int64
# Gender             object
# Age                 int64
# EstimatedSalary     int64
# Purchased           int64
# =============================================================================

data_copy = data.copy()

data.drop(columns=['User ID'], axis=1, inplace=True)

X = data.iloc[:, :3]
y = data.iloc[:, 3]

from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
X.iloc[:, :1] = encoder.fit_transform(X.iloc[:, :1])

scaler = StandardScaler()
X.iloc[:, 2:3] = scaler.fit_transform(X.iloc[:, 2:3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model = model.fit(X_train, y_train) 

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix as cm
matrix = cm(y_test, y_pred)
matrix
# =============================================================================
# array([[66,  2],
#        [ 9, 23]], dtype=int64)
# =============================================================================
