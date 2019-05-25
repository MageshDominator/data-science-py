# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:55:11 2019

@author: MAGESHWARAN
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# --------------------Data Preprocessing------------------------------
data = pd.read_csv("titanic_data.csv")

# Drop irrelavant features
processed_data = data.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis = 1)

# Fillna with mean of ages and for embarked fillna with most repeated class
processed_data["Age"] = processed_data["Age"].fillna(processed_data["Age"].mean())
processed_data["Embarked"] = processed_data["Embarked"].fillna(processed_data["Embarked"].value_counts().idxmax())

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse = False)

features = ["Sex", "Embarked"]

# Label encoding for binary class and One hot encoding for multiclass
for feature in features:
    processed_data[feature] = label_encoder.fit_transform(processed_data[feature])
    # convert 1-D array to 2-D for one hot encoding
    temp_matrix = processed_data[feature].values
    temp_matrix = temp_matrix.reshape(-1, 1)
    store = one_hot_encoder.fit_transform(temp_matrix)
    # creating new column indexes
    index_ = [feature + "_" + str(i) for i in range(len(store[0]))]
    store_df = pd.DataFrame(store, columns=index_)
    # concat the encoded features to DataFrame
    processed_data = pd.concat([processed_data, store_df], axis=1)

processed_data = processed_data.drop(features, axis=1)

# Feature Scaling
scaler = StandardScaler()
processed_data["Age"] = scaler.fit_transform(processed_data["Age"].values.reshape(-1, 1))
processed_data["Fare"] = scaler.fit_transform(processed_data["Fare"].values.reshape(-1, 1))

X = processed_data.iloc[:, 1:].values
y = processed_data.iloc[:, 0].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# ------------------------Training and Prediction-------------------
# ------------------------Random Forest Classifier------------------
model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
