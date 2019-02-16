# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:19:40 2019

@author: MAGESHWARAN
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------------Data Preprocessing-----------------------------
data = pd.read_csv("50_Startups.csv")

X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

# -------------------One Hot Encoding-------------------------------
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])

one_hot_encoder = OneHotEncoder()
temp = one_hot_encoder.fit_transform((X[:, 3].reshape(-1, 1))).toarray()

# remove one feature from temp(Handling Dummy variable trap)
X = np.concatenate((X[:, :2], temp[:, :2]), axis=1)

# Feature scaling is not needed for multiple Linear Regression, model takes care of it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# -----------------Training and Prediction----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Score", r2_score(y_test, y_pred) * 100 , "%")