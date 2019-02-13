# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:54:11 2019

@author: MAGESHWARAN
"""

import scipy.io
import pandas as pd
import configparser
import os

# reading config.inf file
config = configparser.ConfigParser()
config.read("config.inf")

section = config.sections()

# storing section names into a list
path = config.options(section[0])

# Extracting i/o filepath using config parser
input_path = config.get(section[0], path[0])
output_path = config.get(section[0], path[1])
input_files = []
output_files = []

# walk through the path and identify ".mat" files
for root, _, filenames in os.walk(input_path):
    # print(root)
    for filename in filenames:
        if filename.endswith(".mat"):
            pathiter = os.path.join(root, filename)
            input_files.append(pathiter)
            out_file = os.path.join(output_path, filename)
            newname =  out_file.replace('.mat', '.csv')
            output_files.append(newname)

print(output_files)
for i in range(len(input_files)):
    input_file = input_files[i]
    output_file = output_files[i]

    # stored as a dictionary
    mat = scipy.io.loadmat(input_file)

    print(mat.keys())

    # If not possible to read the .mat file in your system,
    # ------------ Manually Do the following -----------
    # 1. Get the matrix names through the above print statement
    # 2. modify x_in and y_out with respecting matrix_names (as string)
    # Example: x_in = "X" and y_out = "y"
    # ------------------------------------------------------

    # convert it to a list
    X = mat["X"].tolist()
    y = mat["y"].tolist()

    column_names = []

    # add all the features and output in single list
    for i in range(len(X)):
        X[i].append(y[i][0])

    # creating namelist for features and output columns
    for i in range(len(X[0])):
        if i < len(X[0]) - 1:
            column_names.append("X" + str(i))
        else:
            column_names.append("Y")

    # covert that to a pandas dataframe
    data = pd.DataFrame(X, columns = column_names)

    # Save the dataframe as a CSV file
    data.to_csv(output_file, sep = ",")