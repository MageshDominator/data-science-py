# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:17:46 2019

@author: MAGESHWARAN
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv("Student-data.csv")

# ---------------Feature Extraction and Selection--------------------
# copying only categorical object(nominal) variables
nominal_data = data.select_dtypes(include=['object']).copy()
nominal_data = pd.concat([nominal_data, data["Medu"], data["Fedu"], data["traveltime"], data["studytime"], data["failures"], data["famrel"], data["freetime"], data["goout"], data["Dalc"], data["Walc"], data["health"]], axis = 1)
# understand the column categories and description
# =============================================================================
# col = nominal_data.columns
# for feature in col:
#     print(nominal_data[feature].value_counts())
#
# =============================================================================

# binary categories, directly use Label encoder
y_n_features = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "address", "internet", "romantic", "passed", "famsize", "school", "sex", "Pstatus"]

label_encoder = LabelEncoder()
for feature in y_n_features:
    nominal_data[feature] = label_encoder.fit_transform(nominal_data[feature])

# features with multiple categories
one_hot_features = ["Mjob", "Fjob", "reason", "guardian", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health"]

one_hot_encoder = OneHotEncoder(sparse = False)
for feature in one_hot_features:
    # first do label encoding
    nominal_data[feature] = label_encoder.fit_transform(nominal_data[feature])
    # convert 1-D array to 2-D for one hot encoding
    temp_matrix = nominal_data[feature].values
    temp_matrix = temp_matrix.reshape(-1, 1)
    store = one_hot_encoder.fit_transform(temp_matrix)
    # creating new column indexes
    index_ = [feature + "_" + str(i) for i in range(len(store[0]))]
    store_df = pd.DataFrame(store, columns = index_)
    # concat the encoded features to DataFrame
    nominal_data = pd.concat([nominal_data, store_df], axis = 1)

# Drop base features(one hot encoded and also output class)
new_data = nominal_data.drop(["Mjob", "Fjob", "reason", "guardian", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "passed"], axis = 1)

# include numerical data from the dataset
new_data = pd.concat([new_data, data["age"], data["absences"]], axis = 1)
X = new_data.iloc[:, : ].values
# 17 is the index value of "passed"(output_class)
y = nominal_data.iloc[:, 17].values

# ------------------Feature Scaling--------------------------------
scaler = StandardScaler()
X= scaler.fit_transform(X)

# ------------------Model Selection--------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 11)

# -----------------Training and Prediction------------------------
svm_model = SVC(kernel = "rbf")
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)


print("Accuracy score:", accuracy_score(y_test, y_pred) * 100, "%")

from sklearn.model_selection import GridSearchCV
parameters = [{"kernel": ["rbf"], "C": [0.5, 1, 10]}]
classifier = GridSearchCV(estimator=svm_model, param_grid=parameters, n_jobs=2, cv=5)
classifier.fit(X_train, y_train)
print(classifier.cv_results_.keys())
accuarcy_score = classifier.best_score_
best_params = classifier.best_params_
