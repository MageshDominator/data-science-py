# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:44:41 2019

@author: MAGESHWARAN
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# ---------------------DATA PreProcessing------------------------
data = pd.read_csv("IBM_HR.csv")
data = data.drop(["EmployeeCount", "EmployeeNumber",
                  "StandardHours", "Over18"], axis=1)

nominal_data = data.select_dtypes(include=['object']).copy()
nominal_data = pd.concat([nominal_data, data[["Education", "JobInvolvement",
                                              "JobLevel",
                                              "EnvironmentSatisfaction",
                                              "JobSatisfaction",
                                              "PerformanceRating",
                                              "RelationshipSatisfaction",
                                              "StockOptionLevel",
                                              "TrainingTimesLastYear",
                                              "WorkLifeBalance"]]], axis=1)

cols = nominal_data.columns

# Label and OneHot encoding
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)

# ================TO VIEW and categorize BINARY and MULTI-CLASS==============
#
# for col in cols:
#     print(nominal_data[col].value_counts())
#     input("Press Enter to continue...")
#
# ===========================================================================
binary_class = ["Attrition", "Gender", "OverTime"]
multi_class = ["BusinessTravel", "Department", "EducationField",
               "JobRole", "MaritalStatus",
               "Education", "JobInvolvement",
               "JobLevel", "EnvironmentSatisfaction", "JobSatisfaction",
               "PerformanceRating", "RelationshipSatisfaction",
               "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance"]

# ---------------Label Encoding for Binary class features--------------------
for feature in binary_class:
    nominal_data[feature] = label_encoder.fit_transform(nominal_data[feature])

# ---------------Label Encoding for multi class features---------------------
for feature in multi_class:
    # use label encoder
    nominal_data[feature] = label_encoder.fit_transform(nominal_data[feature])
    # convert 1-D array to 2-D
    temp = nominal_data[feature].values
    temp = temp.reshape(-1, 1)
    # use one hot encoder
    store = one_hot_encoder.fit_transform(temp)
    # create new column indexes for each class
    index_ = [feature + "_" + str(i) for i in range(len(store[0]))]
    store_df = pd.DataFrame(store, columns=index_)
    # concat new one hot encoded features to dataframe
    nominal_data = pd.concat([nominal_data, store_df], axis=1)

# Remove the features with base classes
nominal_data = nominal_data.drop(multi_class, axis=1)

# duplicate quantized features
quantized_data = data[["Age", "DailyRate", "DistanceFromHome", "HourlyRate",
                       "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
                       "PercentSalaryHike", "TotalWorkingYears",
                       "YearsAtCompany", "YearsInCurrentRole",
                       "YearsSinceLastPromotion", "YearsWithCurrManager"]]
processed_data = pd.concat([nominal_data, quantized_data], axis=1)

# Input and Output matrices
X = processed_data.iloc[:, 1:].values
y = processed_data.iloc[:, 0].values

# ------------------------Feature Scaling-----------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------Cross Validation-----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)

# -----------------------Training and Prediction----------------------------
svc_model = SVC(kernel="rbf", random_state=0)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

print("Model Accuracy:", f1_score(y_test, y_pred_svc) * 100, "%")

# -----------------------Logistic Regression--------------------------------
Log_model = LogisticRegression(random_state=0)
Log_model.fit(X_train, y_train)
y_pred_Log = Log_model.predict(X_test)

print("Model Accuracy:", f1_score(y_test, y_pred_Log) * 100, "%")

# ---------------------MLP Classifier-------------------------------------
MLP_model = MLPClassifier(activation="logistic", alpha=0.01, solver="lbfgs",
                          learning_rate="adaptive", random_state=0)
MLP_model.fit(X_train, y_train)
y_pred_MLP = MLP_model.predict(X_test)

print("Model Accuracy:", f1_score(y_test, y_pred_MLP) * 100, "%")
