# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:13:24 2019

@author: MAGESHWARAN
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Position_salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Adding polynomial degrees to features in X
polynomial_features = PolynomialFeatures(degree = 4)
X_poly = polynomial_features.fit_transform(X)

# -----------------------Training------------------------------
linear_model = LinearRegression()
linear_model.fit(X_poly, y)

# --------------Visualization of Polynomial Regression--------------
plt.scatter(X, y, color = "green")
plt.plot(X, linear_model.predict(polynomial_features.fit_transform(X)), color = "red")
plt.title("Polynomial Regression to Predict Salary")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# --------------------Prediction------------------------------
linear_model.predict(polynomial_features.fit_transform(6.5))
