# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:49:36 2019

@author: MAGESHWARAN
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
# import keras
from keras.models import Sequential
from keras.layers import Dense

# -----------------------Data PreProcessing------------------------------------
data = pd.read_csv("Churn_Modelling.csv")

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Geography"] = data["Geography"].map({"France": 0, "Germany": 1, "Spain": 2})

MultiCat_features = ["Tenure", "NumOfProducts"]

# label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)

# ----------------One Hot encoding for MultiClass features---------------------
for feature in MultiCat_features:
    temp = data[feature].values
    temp = temp.reshape(-1, 1)
    store = one_hot_encoder.fit_transform(temp)

    index_ = [feature + "_" + str(i) for i in range(len(store[0]))]

    store_df = pd.DataFrame(store, columns=index_)

    data = pd.concat([data, store_df], axis=1)

data = data.drop(MultiCat_features, axis=1)

X = data.drop(["Exited"], axis=1).values
y = data["Exited"].values

# ---------------------------Feature Scaling-----------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------Model Selection-----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# ----------------------Training and Prediction--------------------------------
sklearn_model = MLPClassifier()
sklearn_model.fit(X_train, y_train)

y_pred_sklearn = sklearn_model.predict(X_test)

print("sklearn Model Accuracy:", f1_score(y_test, y_pred_sklearn) * 100, "%")

# -------------------------ANN with Keras--------------------------------------
model = Sequential()
model.add(Dense(output_dim=12, kernel_initializer='glorot_uniform',
                activation="relu", input_dim=23))

model.add(Dense(output_dim=12, kernel_initializer='glorot_uniform',
                activation="relu"))

model.add(Dense(output_dim=1, kernel_initializer='glorot_uniform',
                activation="sigmoid"))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
