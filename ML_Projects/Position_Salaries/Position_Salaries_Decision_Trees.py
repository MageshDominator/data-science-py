# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:14:29 2019

@author: MAGESHWARAN
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------Data Preprocessing--------------------------
data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values


# ----------------Training and Prediction-----------------------
model = DecisionTreeRegressor()
model.fit(X, y)

y_pred = model.predict(6.5)

# ----------------Visualization--------------------------------
plt.scatter(X, y, color = "red")
plt.plot(X, model.predict(X), color = "green")
plt.title("Decision Tree Regressor")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()