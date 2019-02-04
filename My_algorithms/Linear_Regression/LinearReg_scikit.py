# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:21:40 2019

@author: MAGESHWARAN
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


data = pd.read_csv("ex1data2.csv", sep = ",")
# plt.matshow(data.corr())

# no. of examples
m = data.shape[0]

# no. of features
n = data.shape[1] - 1

X = np.zeros((m, n + 1))
y = np.array((m, 1))

# load data into X
X[:, 1] = data["X0"].values
X[:, 2] = data["X1"].values

# Load data into Y
y = data["Y"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# creating object for the linear model
linear_model = LinearRegression()

# fit data into the model
linear_model.fit(X_train, y_train)

# run prediction for the test set
predicted_output = linear_model.predict(X_test)
print(predicted_output)
print("Model score:", r2_score(y_test, predicted_output))