
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:15:34 2018

@author: Andre
"""
#not all these imports are needed, i think you just need json and pandas
import json
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
from pandas.io.json import json_normalize

#take data from only one.json file and put it in a DataFrame
data = []
with open("domoticz-2018-03-19.json") as fp:
    for line in fp:
        data.append(json.loads(line))
    df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
print(df.head(20))

#!!!!!!!!!!!!
#read from multiple json files in a folder and parse the data
data = [] #list to store data
path = 'C:/Users/Andre/.spyder-py3/JSON_Files' #make a folder and put all json files u want to use in it.
                                               #This value should be the path to that folder
                                                #NOTE: the .py file and the .json files need to be in the same location
                                                #I would make a new folder, write the .py file in that folder, and only store it and json files in the folder
listing = os.listdir(path)
for infile in listing: #for every file in this folder
    if infile.endswith(".json"): #if it's a json file
        for line in open(infile): #read every line of the file
            data.append(json.loads(line)) #put every line into the list
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns') #put data from list into a DataFrame
#look on manipData2.py for other methods on how to change data in the DataFrame

#format data

#choose certain columns
df = df[['_id','_source.message']]

 #convert _id column to datetime
df._id = pd.to_datetime(df._id)

 #show only time from date time
df['time'] = [d.time() for d in df['_id']]

 #show only date from date time
df['dates'] = [d.time() for d in df['_id']]

#sort values by specified column
df = df.sort_values(by=['_id'])

#describe our dataset
print(df.shape)
print(df.dtypes)
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None)
pd.DataFrame([range(10)])
print(df.describe())

#Trimming columns without important data
df.drop('_source.priority', axis=1, inplace=True)
df.drop('_score', axis=1, inplace=True)
df.drop('_type', axis=1, inplace=True)

#Moving misplaced data
numRows = df.shape[0]
index = 0
while index<numRows:
    if (df.loc[index, '_source.MESSAGE'] != 'nan'):
        misplacedData = df.loc[index, '_source.MESSAGE']
        misplacedData1 = df.loc[index, '_source.SUBJECT']
        df.at[index, '_source.message'] = misplacedData + ' ' + misplacedData1
    index+=1
    
#Trims dataframe of columns just copied
df.drop('_source.MESSAGE', axis=1, inplace=True)
df.drop('_source.PRIORITY', axis=1, inplace=True)
df.drop('_source.SUBJECT', axis=1, inplace=True)
