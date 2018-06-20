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

#!!!!!!!!!!!!
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
            

print(count)

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
