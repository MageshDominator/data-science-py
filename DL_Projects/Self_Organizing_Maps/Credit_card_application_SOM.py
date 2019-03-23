# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:03:08 2019

Creator: Mageshwaran
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# ----------------------Data Preprocessing-------------------------------------
data = pd.read_csv("Credit_Card_Applications.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ---------------------Feature Scaling-----------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# ----------------Self Organizing maps implementation--------------------------
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)


# ---------------Visualizing SOM using pylab-----------------------------------
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o", "s"]
colors = ["r", "g"]

for i, x in enumerate(X):
    # getting the winning node for each sample    
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]],
         markersize=10, markeredgewidth=2, markerfacecolor="None")

show()

# ---------------------------Fraud detection-----------------------------------
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2, 3)], mappings[(2, 4)], mappings[(2, 5)]), axis=0)
frauds = scaler.inverse_transform(frauds)