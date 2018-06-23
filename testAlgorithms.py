# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:22:06 2018
@author: maxjf, partially using Andrew's code from jsonToData.py
"""
import json
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from pandas.io.json import json_normalize
import os
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#!!!!!!!!!!!!
#read from multiple json files in a folder and parse the data
data = [] #list to store data
path = 'C:/Users/Andre/.spyder-py3/JSON_Files/ifttt/domo' #make a folder and put all json files u want to use in it.
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

#choose certain columns
df = df[['_id','_source.message']]

#find the size of the column and store it
#print(df.shape)
size = df.shape
numRows = size[0]
#print(numRows)
uniqueMessageList = []
#df.head(100)

for x in range(numRows):
    message = df.iloc[x,1]
    if x == 0:
        uniqueMessageList.append(message)
    else:
        flag = True;
        for r in range(len(uniqueMessageList)):
            if message == uniqueMessageList[r]:
                flag = False;
        if flag:
            uniqueMessageList.append(message)

"""for i in range(len(uniqueMessageList)):
    print(uniqueMessageList[i])"""

#print(len(uniqueMessageList))
sensorType = list(range(numRows))
sensorState = list(range(numRows))
df['Sensor Type'] = sensorType
df['Sensor State'] = sensorState
count = 0
successEdit = []
for x in range(numRows):
    message = df.iloc[x,1]
    #print(message)
    #print(type(message))
    if str(message).find('sensor-motion-1 Switch') > -1:
        df['Sensor Type'][x] = 'Motion Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'  
    elif str(message).find('sensor-light-1') > -1:
        df['Sensor Type'][x] = 'Light Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'  
    elif str(message).find('Interior Motion - Room 1') > -1:
        df['Sensor Type'][x] =  'Motion Sensor Room 1'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'  
    elif str(message).find('Interior Motion - Room 2') > -1:
        df['Sensor Type'][x] = 'Motion Sensor Room 2'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'  
    elif str(message).find('Smart Plug (AP)') > -1:
        df['Sensor Type'][x] = 'Smart Plug (AP)'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'
    elif str(message).find('Lamp') > -1:
        df['Sensor Type'][x] = 'Lamp'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'
        else:
            df['Sensor State'][x] = 'On'
    elif str(message).find('Lock1') > -1:
        df['Sensor Type'][x] = 'Lock 1'
        count += 1
        if message.lower().find('unlocked') > -1:
            df['Sensor State'][x] = 'Unlocked'
        else:
            df['Sensor State'][x] = 'Locked'
    elif str(message).find('Domoticz test message!') > -1:
        df['Sensor Type'][x] = 'Not applicable'
        df['Sensor State'][x] = 'Not applicable'
        count += 1
    elif str(message).find('Plug (Wink') > -1:
        df['Sensor Type'][x] = 'Plug (Wink)'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'        
        else:
            df['Sensor State'][x] = 'On'
    elif str(message).find('nan') > -1:
        df['Sensor Type'][x] = 'Not applicable'
        df['Sensor State'][x] = 'Not applicable'
        count += 1                      
    elif str(message).find('Hallway1') > -1:
        df['Sensor Type'][x] = 'Hallway1 Motion'
        count += 1
        if message.lower().find('off') > -1:
            df['Sensor State'][x] = 'Off'        
        else:
            df['Sensor State'][x] = 'On'
    elif str(message).find('MotionSensorRoom1') > -1:
        df['Sensor Type'][x] = 'Motion Sensor Room 1'
        count += 1
        if message.lower().find('Motion Detected') > -1:
            df['Sensor State'][x] = 'On'

#format data
df._id = pd.to_datetime(df._id) #turn _id column to a datetime object
df['intrusion'] = [0] * df.shape[0] #adds intrusion data
df = df[['_id', 'Sensor Type', 'Sensor State', 'intrusion']] #selects rows to be used

data_ifttt = [] #array for parsing ifttt data
def parseifttt(d): #method for reading json files
    path = 'C:/Users/Andre/.spyder-py3/JSON_Files/ifttt/ifttt'
    listing = os.listdir(path)
    for infile in listing:
        if infile.endswith(".json"):
            for line in open(infile):
                d.append(json.loads(line))
    return d
data_ifttt = parseifttt(data_ifttt)
df_ifttt = pd.DataFrame.from_dict(json_normalize(data_ifttt), orient='columns')
df_ifttt['intrusion'] = [0] * df_ifttt.shape[0] #adds intrusion column

#removes NaN values and enters appropriate data
n = 0
for i in df_ifttt["_source.message"].isnull():
    if i:
        df_ifttt["_source.message"][n] = df_ifttt["_source.status"][n]
    n = n + 1

#format the data
df_ifttt = df_ifttt[['_id','_source.message', '_source.device', 'intrusion']]
df_ifttt._id = pd.to_datetime(df_ifttt._id)
df_ifttt['Sensor Type'] = df_ifttt['_source.device']
df_ifttt['Sensor State'] = df_ifttt['_source.message']
df_ifttt['dates'] = [d.date() for d in df_ifttt['_id']]

#if data was recorded during intrusion simulation: enter a one in the intrusion column
n = 0
for i in df_ifttt['dates']:
    if i.strftime('%Y-%m-%d') == "2018-04-18":
        df_ifttt['intrusion'][n] = 1
    n = n + 1

df_ifttt = df_ifttt[['_id', 'Sensor Type', 'Sensor State', 'intrusion']]
df_ifttt = df_ifttt.sort_values(by=['Sensor Type', '_id'])
df = df.append(df_ifttt, ignore_index=True) #append the data
df = df[pd.notnull(df['Sensor State'])]
df = df[pd.notnull(df['Sensor Type'])]

#binarize data
X = [str(d) for d in df['Sensor Type']]
Y = [str(d) for d in df['Sensor State']]
def binarize(Z):
    encoded = preprocessing.LabelEncoder()
    encoded.fit(Z)
    newEncode = encoded.transform(Z)
    newEncode = newEncode.reshape(-1, 1)
    #One Hot Encoder takes the enumerated data and turns it to binary
    OHE = preprocessing.OneHotEncoder(sparse=False)
    #print (OHE.fit_transform(newEncode))
    Z = OHE.fit_transform(newEncode)
    return Z
X = binarize(X)
Y = binarize(Y)

#put binarized data into DataFrames
columnsType=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
types = pd.DataFrame(X, columns=columnsType)
columnsState=['1', '2', '3', '4', '5', '6', '7', '8']
states = pd.DataFrame(Y, columns=columnsState)

#set columns for time
#df_ifttt['month'] = [4] * df_ifttt.shape[0]
#df_ifttt['day'] = [0] * df_ifttt.shape[0]
#df_ifttt['hour'] = [0] * df_ifttt.shape[0]
#df_ifttt['minute'] = [0] * df_ifttt.shape[0]
#for d in df_ifttt.index:
#    m, day, h, mini = df_ifttt['_id'][d].strftime('%m'), df_ifttt['_id'][d].strftime('%d'), df_ifttt['_id'][d].strftime('%H'), df_ifttt['_id'][d].strftime('%M')
#    i1 = int(m)
#    i2 = int(day)
#    i3 = int(h)
#    i4 = int(mini)
#    df_ifttt['month'][d] = i1
#    df_ifttt['day'][d] = i2
#    df_ifttt['hour'][d] = i3
#    df_ifttt['minute'][d] = i4

#append the binarized data to the main array (also removes 'Sensor State' and 'Sensor Type)
df = df[['intrusion']] #note: for this build, column "_id" was removed for testing purposes.
df = df.join(types)
df = df.join(states)

print(df.head(20))

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option('date_yearfirst', True)
pd.set_option('display.max_rows', 2500)
"""swx-u-range-sensor-motion-1 Switch >> OFF (Resolved)

swx-u-range-sensor-light-1 >> OFF (Resolved)

swx-u-range-sensor-light-1 >> ON (Resolved)

swx-u-range-sensor-motion-1 Switch >> ON (Resolved)

Alarm Off - Interior Motion - Room 1 (Resolved)

Alarm - Interior Motion - Room 2 (Resolved)

Alarm - Interior Motion

Alarm Off

Alarm - Interior Motion - Room 1 (Resolved)

nan (Resolved)

swx-u-range-sensor-motion-1 Alarm type: 0x07 AC >> OFF

Alarm Off - Interior Motion - Room 2 (Resolved)

Domoticz test message! (Resolved)

nan (Resolved)

Sensor swx-u-range-sensor-light-1 Last Update: 2018-03-19 18:11:22 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-14 10:52:04 [!= 1 minutes]

Plug (Wink Turned On (Resolved)

MotionSensorRoom1 - Motion Detected (Resolved)

Smart Plug (AP) Turned On (Resolved)

Hallway1 - Motion Detected (Resolved)

Hallway1 - Alarm Off (Resolved)

Lock1 Unlocked (Resolved)

MotionSensorRoom1 - Monitoring

Lock1 Locked (Resolved)

Plug (Wink Turned Off (Resolved)

Lamp Turned On (Resolved)

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-17 18:11:20 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-18 06:51:50 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-18 19:51:31 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-19 07:17:48 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-19 18:11:41 [!= 1 minutes]

Sensor swx-u-range-sensor-light-1 Last Update: 2018-04-20 06:49:16 [!= 1 minutes]"""