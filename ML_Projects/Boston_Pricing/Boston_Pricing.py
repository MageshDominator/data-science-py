# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:51:51 2019

@author: MAGESHWARAN
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#-------------Feature Extraction-------------------------
data_train = pd.read_csv("Boston_housing_price.csv")

X_unscaled = data_train.drop(["ID", "Price"], axis = 1)
y = data_train["Price"]

# -----------Feature Scaling----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X_unscaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

# ------------Training and Prediction-------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Test accuracy(R2_score):", r2_score(y_test, y_pred) * 100, "%")

# ------------Vistualization of Predicted vs Actual outputs-------
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")