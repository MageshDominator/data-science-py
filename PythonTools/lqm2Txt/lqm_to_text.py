# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:38:51 2018

@author: MAGESHWARAN
"""
import os
import io
import re
import configparser

# To get input and output files path
config = configparser.ConfigParser()
config_file = config.read(r"config.ini")
input_path = config["FILEPATH"]["INPUT"]

# os.chdir("D:")
for root, _, filenames in os.walk(input_path):
    # print(root)
    for filename in filenames:
        print(filename)
        pathiter = os.path.join(root, filename)
        # print(pathiter)
        newname =  pathiter.replace('.jlqm', '.txt')
        if newname != pathiter:
                os.rename(pathiter,newname)

for root, _, filenames in os.walk(input_path):
    # print(root)
    # print(filenames)
    for filename in filenames:
        if ".txt" in filename:
            with io.open(input_path + filename, "r", encoding="utf-8") as a:
                data = a.read()
                # print(type(data))
                strt = re.search("DescRaw", data).start()
                stop = re.search('"Height":', data).start()
                # print(strt, stop)
                memo_text = data[strt+11:stop-2]
                memo = memo_text.replace(r"\n", "")
                with io.open(input_path + filename, "w", encoding="utf-8") as b:
                    b.write(memo)
