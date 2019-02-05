# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:49:45 2019

@author: MAGESHWARAN
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("Iris.data.csv", sep = ",")

# no. of examples
m = iris.shape[0]

# no. of features
n = iris.shape[1] - 1


X = np.ones((m,n + 1))
y = np.array((m,1))

X[:,1] = iris['X0'].values
X[:,2] = iris['X1'].values
X[:,3] = iris['X2'].values
X[:,4] = iris['X3'].values

#Labels
y = iris['Y'].values

#Mean normalization
for j in range(n):
    X[:, j] = (X[:, j] - X[:,j].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print("Test Accuracy:", accuracy_score(y_pred, y_test) * 100, "%")