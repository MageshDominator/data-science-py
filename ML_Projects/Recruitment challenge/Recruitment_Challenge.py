# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:14:34 2019

@author: MAGESHWARAN
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

facial = pd.read_csv("facial_similarity_reports.csv")
facial = facial.drop(["Unnamed: 0", "created_at", "properties",
                      "attempt_id"], axis=1)
facial = facial.dropna()
doc = pd.read_csv("doc_reports.csv")
doc = doc.drop(["Unnamed: 0", "created_at", "properties",
                "attempt_id", "conclusive_document_quality_result",
                "conclusive_document_quality_result",
                "colour_picture_result", "data_comparison_result",
                "compromised_document_result",
                "data_consistency_result"], axis=1)
doc = doc.dropna()
data = pd.merge(facial, doc, how="left", on=["user_id"])
data = data.drop(["image_quality_result", "supported_document_result"], axis=1)
data = data.drop_duplicates(["user_id"])
data = data.drop(["user_id"], axis=1)
data = data.dropna()
desc = data.describe()

binary_data = data.drop(["sub_result"], axis=1)
multi_data = data["sub_result"]
columns = binary_data.columns

for col in columns:
    binary_data[col] = binary_data[col].map({"clear": 1, "consider": 0})

multi_data = multi_data.map({"clear": 1, "caution": 2, "suspected": 3})

processed_data = pd.concat([binary_data, multi_data], axis=1)

scaler = StandardScaler()
X = processed_data.drop(["result_x", "result_y"], axis=1).values
y = processed_data[["result_x", "result_y"]].values

# ------------------------Feature Scaling--------------------------------
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# -----------------------Training and Prediction-------------------------
svm_model_facial = SVC(random_state=0)
svm_model_facial.fit(X_train, y_train[:, 0])
y_pred_facial = svm_model_facial.predict(X_test)

print("Model accuracy for Facial Result:",
      f1_score(y_test[:, 0], y_pred_facial) * 100, "%")

svm_model_doc = SVC(random_state=0)
svm_model_doc.fit(X_train, y_train[:, 1])
y_pred_doc = svm_model_doc.predict(X_test)

print("Model accuracy for Doc Result:",
      f1_score(y_test[:, 1], y_pred_facial) * 100, "%")
