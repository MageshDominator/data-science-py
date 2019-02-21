# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:30:57 2019

@author: MAGESHWARAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ------------------Data Preprocessing--------------------------
data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# -----------------Feature Scaling------------------------------
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# ----------------Training and Prediction-----------------------
model = SVR()
model.fit(X_scaled, y_scaled)

y_pred = scaler_y.inverse_transform(model.predict(scaler_x.transform(np.array([[6.5]]))))

# ----------------Visualization--------------------------------
plt.scatter(X_scaled, y_scaled, color = "red")
plt.plot(X_scaled, model.predict(X_scaled), color = "green")
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()