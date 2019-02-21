# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:33:14 2019

@author: MAGESHWARAN
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------Data Preprocessing--------------------
data =  pd.read_csv("Social_Network_ads.csv")

X = data.iloc[:, 1:4].values
y = data.iloc[:, 4].values

encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# ---------------------Training and Prediction---------------
model = RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Accuracy score:", accuracy_score(y_test, y_pred) * 100, "%")