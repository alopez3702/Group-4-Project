
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:57:39 2018

@author: Andre
"""

import json
import csv
import pandas
import scipy
import numpy

#read the .json file; put contents in a list
data = []
with open("domoticz-2018-03-19.json") as f:
    s = json.loads(f)
    for line in f:
        data.append(json.loads(line))

#make .csv file using contents from the new list
jtc = open('domoticz-2018-03-19.csv', 'w')
csvwriter = csv.writer(jtc)
count = 0
for d in data:
   if count == 0:
       header = d.keys()
       csvwriter.writerow(header)
       count += 1
   csvwriter.writerow(d.values())
jtc.close()

#reads .csv and puts it into a DataFrame
        #NOTE: One problem I got when reading a .csv file was that the first row just repeated the column names, and obstructed the data.
        #But, I found some ways to overcome this issue, simply by deleting the row
c = "domoticz-2018-03-20.csv" #this .csv file I got by getting "domoticz-2018-03-20.json.bz2" from GitHub Safehouse data, unzipping it, and using an online converter to turn it into a .csv
names = ['_index', '_tpye', '_id', '_score', '_source__subject', '_source__status', '_source__user', '_source__Time', '_source__message', '_source__priority']
dataset = pandas.read_csv(c, names=names)

#Puts DataFrame into an array
array = dataset.values
array = numpy.delete(array, (0), axis=0) #deletes first row in array, since in this example the first row repeats the column names
#print(array)

#makes a new DataFrame with only certain columns of previous DataFrame
#just in case we can only work with certain columns
ds = dataset[['_index','_source__message']] #takes the columns that will be used in the new DataFrame
#print(ds.head(5))

#removes rows from a DataFrame
#just in case certain rows need to be removed from the data
dataset = dataset[dataset._source__message != '_source__message'] #removes rows where in the selected column (dataset._source__message) the value is the selected value
#print(dataset.head(5))

#handling non-numeric data
#Essentailly, this code snippet enumerates every non-numerical data point.
#Note: Not sure how turning words into numbers affects the data. Does this ruin our data?
#This code is used to make dataset actually usable with machine learning algorithms.
dataset.convert_objects(convert_numeric=True)
dataset.fillna(0, inplace=True)

def handle_non_numeric_data(dataset):
    columns = dataset.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            dataset[column] = list(map(convert_to_int, dataset[column]))
    return dataset

dataset = handle_non_numeric_data(dataset)
print(dataset.head())
