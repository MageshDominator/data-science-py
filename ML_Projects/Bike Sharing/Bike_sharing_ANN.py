# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:51:46 2019

@author: MAGESHWARAN
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# read hours dataset
hour_df = pd.read_csv("hour.csv")

print(hour_df.columns)

# ----------------------- Feature Selection ---------------------
corr_initial = hour_df.corr()
# generate heatmap to for correlation
heat_map2 = sns.heatmap(corr_initial, linewidth=0.01)
# plt.savefig("heat_map2")

# create features from category variables, making binary dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

for field in dummy_fields:
    dummies = pd.get_dummies(hour_df[field], prefix = field, drop_first = False)
    hour_df = pd.concat([hour_df, dummies], axis = 1)

# drop features
drop_fields = ["instant", "dteday", "season", "yr", "mnth", "weekday", "weathersit", "workingday", "hr", "atemp"]

data = hour_df.drop(drop_fields, axis = 1)

print(data.columns)
corr = data.corr()

# generate heatmap to for correlation
heat_map = sns.heatmap(corr, linewidth=0.01)
# plt.savefig("heat_map")

# ------------------------ Feature Scaling --------------------
# scale all the quantitative features
quant_features = ["temp", "casual", "registered", "cnt", "hum", "windspeed"]

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for feature in quant_features:
    mean, std = data[feature].mean(), data[feature].std()
    scaled_features[feature] = [mean, std] # used for later purpose
    # making features with 0 mean and std of 1
    data.loc[:, feature] = (data[feature] - mean) / std

X = data.drop(["cnt", "casual", "registered"], axis = 1)
y = data[["cnt", "casual", "registered"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.002)

# --------------------- Training and Prediction--------------------
linear_model = MLPRegressor()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

print("Test score(R2):", r2_score(y_pred, y_test) * 100, "%")

print("Test score(model_score):", linear_model.score(X_test, y_test) * 100, "%")