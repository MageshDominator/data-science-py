# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:32:51 2019

@author: MAGESHWARAN
"""

import csv
import configparser

# To get input and output files path
config = configparser.ConfigParser()
config_file = config.read(r"config.ini")
input_file = config["FILEPATH"]["INPUT"]
output_file = config["FILEPATH"]["OUTPUT"]

lines = []
values = []

# reading text file and converting to list
with open(input_file, "r") as data:
    for line in data:
        lines.append(line.strip())
    for value in lines:
        values.append(value.split(","))
print(values)

# writing to csv from list of list
with open(output_file, "w") as csvfile:
    # object for writing to csv files
    writer = csv.writer(csvfile)
    n_features = len(values[0])
    features = []
    for i in range(n_features):
        if i == n_features - 1:
            features.append("Y")
        else:
            features.append("X" + str(i))
    writer.writerow(features)
    writer.writerows(values)