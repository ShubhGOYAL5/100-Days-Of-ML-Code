# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 01:46:32 2020

@author: GOYAL
"""

# importing libraries
import pandas as pd
import numpy as np

# loading dataset
data = pd.read_csv('Data.csv')

#data.head()

X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

# handling missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])

# creating a dummy variable
encoder_onehot = OneHotEncoder(categorical_features=[0], handle_unknown='ignore')
X = encoder_onehot.fit_transform(X).toarray()

y = encoder.fit_transform(y)

# splitting the datasets in training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


