# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:36:41 2020

@author: GOYAL
"""

import pandas as pd
import math

data = pd.read_csv('kc_house_data.csv')

data.shape
data.isnull().sum()

# =============================================================================
# id               0
# date             0
# price            0
# bedrooms         0
# bathrooms        0
# sqft_living      0
# sqft_lot         0
# floors           0
# waterfront       0
# view             0
# condition        0
# grade            0
# sqft_above       0
# sqft_basement    0
# yr_built         0
# yr_renovated     0
# zipcode          0
# lat              0
# long             0
# sqft_living15    0
# sqft_lot15       0
# =============================================================================

data.dtypes

# =============================================================================
# id                 int64
# date              object
# price            float64
# bedrooms           int64
# bathrooms        float64
# sqft_living        int64
# sqft_lot           int64
# floors           float64
# waterfront         int64
# view               int64
# condition          int64
# grade              int64
# sqft_above         int64
# sqft_basement      int64
# yr_built           int64
# yr_renovated       int64
# zipcode            int64
# lat              float64
# long             float64
# sqft_living15      int64
# sqft_lot15         int64
# =============================================================================

#data['date2'] = pd.to_datetime(data['date'], format='%Y%m%dT%h%m%s', errors='coerce')
data_copy = data.copy()

data.drop(['id', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'date'], axis=1, inplace=True)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# here rmse: 232422.6236145384


























