# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:54:11 2019

@author: MAGESHWARAN
"""

import scipy.io
import pandas as pd
import configparser

# reading config.inf file
config = configparser.ConfigParser()
config.read("config.inf")

section = config.sections()

# storing section names into a list
path = config.options(section[0])
array = config.options(section[1])
 
# Extracting i/o filepath using config parser
input_file = config.get(section[0], path[0])
output_file = config.get(section[0], path[1])

# stored as a dictionary
mat = scipy.io.loadmat(input_file)

# Extracting matrix names(used in .mat file) through config parser
x_in = config.get(section[1], array[0])
y_out = config.get(section[1], array[1])

print(mat.keys())

# If not possible to read the .mat file in your system,
# ------------ Manually Do the following -----------
# 1. Get the matrix names through the above print statement
# 2. modify x_in and y_out with respecting matrix_names (as string)
# Example: x_in = "X" and y_out = "y"
# ------------------------------------------------------

# convert it to a list
X = mat[x_in].tolist()
y = mat[y_out].tolist()

column_names = []

# add all the features and output in single list
for i in range(len(X)):
    X[i].append(y[i])

# creating namelist for features and output columns
for i in range(len(X[0])):
    if i < len(X[0]) - 1:
        column_names.append("X" + str(i))
    else:
        column_names.append("Y")

# covert that to a pandas dataframe
data = pd.DataFrame(X, columns = column_names)

# Save the dataframe as a CSV file
data.to_csv(output_file, sep = ",", )