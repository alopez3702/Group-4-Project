# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:22:06 2018

@author: maxjf, partially using Andrew's code from jsonToData.py and parseiftttInProgress.py
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


#read from multiple json files in a folder and parse the data
data = [] #list to store data
path = 'C:/Users/maxjf/Summer Internship 2018/domoticz Files' #make a folder and put all json files u want to use in it.
                                               #This value should be the path to that folder
                                                #NOTE: the .py file and the .json files need to be in the same location
                                                #I would make a new folder, write the .py file in that folder, and only store it and json files in the folder
listing = os.listdir(path)
for infile in listing: #for every file in this folder
    if infile.endswith(".json"): #if it's a json file
        for line in open(infile): #read every line of the file
            data.append(json.loads(line)) #put every line into the list
df_domoticz = pd.DataFrame.from_dict(json_normalize(data), orient='columns') #put data from list into a DataFrame
#look on manipData2.py for other methods on how to change data in the DataFrame

#choose certain columns
df_domoticz = df_domoticz[['_id','_source.message']]

#find the size of the column and store it
#print(df_domoticz.shape)
numRows = df_domoticz.shape[0]
#numRows_intr = df_domoticz_intr.shape
#print(numRows)
uniqueMessageList = []
#df_domoticz.head(100)

for x in range(numRows):
    message = df_domoticz.iloc[x,1]
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


df_domoticz['Sensor Type'] = sensorType
df_domoticz['Sensor State'] = sensorState
count = 0
to_delete = []
for x in range(numRows):
    message = df_domoticz.iloc[x,1]
    #print(message)
    #print(type(message))
    if str(message).find('sensor-motion-1 Switch') > -1:
        df_domoticz['Sensor Type'][x] = 'Motion Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'  
    elif str(message).find('sensor-light-1') > -1:
        df_domoticz['Sensor Type'][x] = 'Light Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'  
    elif str(message).find('Interior Motion - Room 1') > -1:
        df_domoticz['Sensor Type'][x] =  'Motion Sensor Room 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'  
    elif str(message).find('Interior Motion - Room 2') > -1:
        df_domoticz['Sensor Type'][x] = 'Motion Sensor Room 2'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'  
    elif str(message).find('Smart Plug (AP)') > -1:
        df_domoticz['Sensor Type'][x] = 'Smart Plug (AP)'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'
    elif str(message).find('Lamp') > -1:
        df_domoticz['Sensor Type'][x] = 'Lamp'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'
        else:
            df_domoticz['Sensor State'][x] = 'On'
    elif str(message).find('Lock1') > -1:
        df_domoticz['Sensor Type'][x] = 'Lock 1'
        count += 1
        if message.lower().find('unlocked') > -1:
            df_domoticz['Sensor State'][x] = 'Unlocked'
        else:
            df_domoticz['Sensor State'][x] = 'Locked'
    elif str(message).find('Plug (Wink') > -1:
        df_domoticz['Sensor Type'][x] = 'Plug (Wink)'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'        
        else:
            df_domoticz['Sensor State'][x] = 'On'                     
    elif str(message).find('Hallway1') > -1:
        df_domoticz['Sensor Type'][x] = 'Hallway1 Motion'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz['Sensor State'][x] = 'Off'        
        else:
            df_domoticz['Sensor State'][x] = 'On'
    elif str(message).find('MotionSensorRoom1') > -1:
        df_domoticz['Sensor Type'][x] = 'Motion Sensor Room 1'
        count += 1
        if message.find('Motion Detected') > -1:
            df_domoticz['Sensor State'][x] = 'Detected'
        else:
            df_domoticz['Sensor State'][x] = 'Monitoring'
    elif str(message).find('Door') > -1:
        df_domoticz['Sensor Type'][x] = 'Door'
        count += 1
        if message.lower().find('unlocked') > -1:
            df_domoticz['Sensor State'][x] = 'Unlocked'
        else:
            df_domoticz['Sensor State'][x] = 'Locked'
    elif str(message).find('Room 1 - Motion Detected') > -1:
        df_domoticz['Sensor Type'][x] = 'Motion Sensor Room 1'
        count += 1
        df_domoticz['Sensor State'][x] = 'On'
    elif str(message).find('Room1 - Alarm Off') > -1:
        df_domoticz['Sensor Type'][x] = 'Motion Sensor Room 1'
        count += 1
        df_domoticz['Sensor State'][x] = 'Off'            
    else: #Any bad data that needs to be thrown out
        to_delete.append(x)
        count += 1             
print(count)
#Delete the bad data (currently the following:
#Alarm - Interior Motion
#Alarm Off
#swx-u-range-sensor-motion-1 Alarm type: 0x07 AC >> OFF
#Domoticz test message!
#nan)
print("Check 0")


print("Check 1")
df_domoticz._id = pd.to_datetime(df_domoticz._id)
#choose certain columns
df_test = df_domoticz[['_id','_source.message']]
 #convert _id column to datetime
df_test._id = pd.to_datetime(df_test._id)
numRows = df_domoticz.shape[0]
print(numRows)
print("Check 2")

 #sort the data in ifttt
df_domoticz['intrusion'] = [0] * df_domoticz.shape[0] #adds intrusion columns
df_domoticz['dates'] = [d.date() for d in df_domoticz['_id']] #adds dates column (used for checking date w/o time)
print("Check 3")
#newCount = 0
for k in range(numRows): #for every date:
    #print(newCount)
    current_date = df_domoticz['dates'][k]
    flag = current_date.strftime('%Y-%m-%d') == "2018-04-18" or current_date.strftime('%Y-%m-%d') =="2018-04-13" or current_date.strftime('%Y-%m-%d') =="2018-04-12"
    if flag: #if it's equal to this date intrusion data was recorded
        df_domoticz['intrusion'][k] = 1 #put a 1 in the intrusion column on the same row
    #newCount += 1        
print("Check 4")        
for k in reversed(range(len(to_delete))):
    df_domoticz.drop(df_domoticz.index[to_delete[k]], inplace=True)
#df_domoticz = df_domoticz[['_id', 'Sensor Type', 'Sensor State', 'intrusion']] #organize dataframe
df_domoticz = df_domoticz.sort_values(by=['Sensor Type', '_id']) #sort data

#6/22/18 notes: It takes about 40 seconds for the code to run
#It takes about 35 seconds for the for loop in between check 3 and 4 to run properly
