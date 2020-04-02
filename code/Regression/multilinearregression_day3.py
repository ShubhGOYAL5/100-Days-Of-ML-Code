# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:38:47 2020

@author: GOYAL
"""

# multiple linear regression

import pandas as pd
import math

data = pd.read_csv('50_Startups.csv')

data.shape
data.isnull().sum()
data.dtypes

import matplotlib.pyplot as plt
plt.scatter(data['R&D Spend'], data['Profit'], color='red')
plt.scatter(data['Administration'], data['Profit'], color='red')
plt.scatter(data['Marketing Spend'], data['Profit'], color='red')

# satisfies the linearity assumption

from statsmodels.stats.outliers_influence import variance_inflation_factor
temp_df = data[['R&D Spend', 'Administration', 'Marketing Spend']]
temp_df['Intercept'] = 1

vif = pd.DataFrame()
vif['variables'] = temp_df.columns
vif['VIF'] = [variance_inflation_factor(temp_df.values, i) for i in range(temp_df.shape[1])]

vif
# threshold set: see if VIF>2.5 ==> some explanatory variables could be highly correlated to each other
# satisfies the multicollinearity assumption

X = data.iloc[:, :4]
y = data.iloc[:, 4]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l_encoder = LabelEncoder()
X[:, 3] = l_encoder.fit_transform(X[:, 3])

oh_encoder = OneHotEncoder(handle_unknown='ignore')
X = oh_encoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

