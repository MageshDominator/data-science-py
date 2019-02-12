# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:32:51 2019

@author: MAGESHWARAN
"""

import csv
import configparser
import os

# To get input and output files path
config = configparser.ConfigParser()
config_file = config.read(r"config.ini")
input_path = config["FILEPATH"]["INPUT"]
output_path = config["FILEPATH"]["OUTPUT"]
seperator = config["FILEPATH"]["SEP"]

# print(seperator)
input_files = []
output_files = []

# os.chdir("D:")
# walk through the path and identify ".txt" files
for root, _, filenames in os.walk(input_path):
    # print(root)
    for filename in filenames:
        if filename.endswith(".txt"):
            pathiter = os.path.join(root, filename)
            input_files.append(pathiter)
            out_file = os.path.join(output_path, filename)
            newname =  out_file.replace('.txt', '.csv')
            output_files.append(newname)

print(output_files)

for i in range(len(input_files)):
    input_file = input_files[i]
    output_file = output_files[i]
    lines = []
    values = []

    # reading text file and converting to list
    with open(input_file, "r") as data:
        for line in data:
            lines.append(line.strip())
        for value in lines:
            values.append(value.split(str(seperator)))
    # print(values)

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