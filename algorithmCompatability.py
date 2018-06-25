# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 01:08:42 2018

@author: Elliott

This file uses the previous code from achieving a data frame from the json.
It enumerates which sensor has gone off and creates a column with the time passed since the last alert
Using the KNN algorithm template
"""

import json
import pandas as pd
from pandas.io.json import json_normalize
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

#read from multiple json files in a folder and parse the data
data = [] #list to store data
path = 'C:/Users/Elliott/Documents/MLVision' #make a folder and put all json files u want to use in it.
                                               #This value should be the path to that folder
                                                #NOTE: the .py file and the .json files need to be in the same location
                                                #I would make a new folder, write the .py file in that folder, and only store it and json files in the folder
listing = os.listdir(path)
for infile in listing: #for every file in this folder
    if infile.endswith(".json"): #if it's a json file
        for line in open(infile): #read every line of the file
            data.append(json.loads(line)) #put every line into the list
df_domoticz = pd.DataFrame.from_dict(json_normalize(data), orient='columns') #put data from list into a DataFrame

#choose certain columns
df_domoticz = df_domoticz[['_id','_source.message']]
#find the size of the column and store it
numRows = df_domoticz.shape[0]

#Optional code to check for unique messages, mainly used in development
"""uniqueMessageList = []
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
            
for i in range(len(uniqueMessageList)):
    print(uniqueMessageList[i])
print(len(uniqueMessageList))"""

sensorType = list(range(numRows))
sensorState = list(range(numRows))
df_domoticz['Sensor Type'] = sensorType
df_domoticz['Sensor State'] = sensorState
count = 0
to_delete = []
for x in range(numRows):
    message = df_domoticz.iloc[x,1]
    if str(message).find('sensor-motion-1 Switch') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Motion Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'  
    elif str(message).find('sensor-light-1') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Light Sensor 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'  
    elif str(message).find('Interior Motion - Room 1') > -1:
        df_domoticz.loc[x, 'Sensor Type'] =  'Motion Sensor Room 1'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'  
    elif str(message).find('Interior Motion - Room 2') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Motion Sensor Room 2'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'  
    elif str(message).find('Smart Plug (AP)') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Smart Plug (AP)'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'
    elif str(message).find('Lamp') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Lamp'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'
    elif str(message).find('Lock1') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Lock 1'
        count += 1
        if message.lower().find('unlocked') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Unlocked'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'Locked'
    elif str(message).find('Plug (Wink') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Plug (Wink)'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'        
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'                     
    elif str(message).find('Hallway1') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Hallway1 Motion'
        count += 1
        if message.lower().find('off') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Off'        
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'On'
    elif str(message).find('MotionSensorRoom1') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Motion Sensor Room 1'
        count += 1
        if message.find('Motion Detected') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Detected'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'Monitoring'
    elif str(message).find('Door') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Door'
        count += 1
        if message.lower().find('unlocked') > -1:
            df_domoticz.loc[x, 'Sensor State'] = 'Unlocked'
        else:
            df_domoticz.loc[x, 'Sensor State'] = 'Locked'
    elif str(message).find('Room 1 - Motion Detected') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Motion Sensor Room 1'
        count += 1
        df_domoticz.loc[x, 'Sensor State'] = 'On'
    elif str(message).find('Room1 - Alarm Off') > -1:
        df_domoticz.loc[x, 'Sensor Type'] = 'Motion Sensor Room 1'
        count += 1
        df_domoticz.loc[x, 'Sensor State'] = 'Off'            
    else: #Any bad data that needs to be thrown out
        to_delete.append(x)
        count += 1             
#print(count) Optional way to check if all rows have been indexed through
print("Check 0")
#Delete the bad data (currently the following:
#Alarm - Interior Motion
#Alarm Off
#swx-u-range-sensor-motion-1 Alarm type: 0x07 AC >> OFF
#Domoticz test message!
#nan)
for k in reversed(range(len(to_delete))):
    df_domoticz.drop(df_domoticz.index[to_delete[k]], inplace=True)
df_domoticz.reset_index(drop = True, inplace = True) #Reset the index for further looping
print("Check 1")

df_domoticz._id = pd.to_datetime(df_domoticz._id)
df_domoticz['dates'] = [d.date() for d in df_domoticz['_id']] #adds dates column (used for checking date w/o time)
print("Check 2")

 #sort the data in ifttt
df_domoticz['intrusion'] = [0] * df_domoticz.shape[0] #adds intrusion columns
print("Check 3")

newCount = 0
numRows = df_domoticz.shape[0]
print("Check 4")

for k in range(numRows): #for every date:
    current_date = df_domoticz['dates'][k]
    flag = current_date.strftime('%Y-%m-%d') == "2018-04-18" or current_date.strftime('%Y-%m-%d') =="2018-04-13" or current_date.strftime('%Y-%m-%d') =="2018-04-12"
    if flag: #if it's equal to this date intrusion data was recorded
        df_domoticz.loc[k,'intrusion'] = 1 #put a 1 in the intrusion column on the same row
    newCount += 1        
print("Check 5")        

df_domoticz = df_domoticz[['_id', 'Sensor Type', 'Sensor State', 'intrusion']] #organize dataframe
df_domoticz = df_domoticz.sort_values(by=['Sensor Type', '_id']) #sort data

#Encodes the categoriacal data of the sensors and records it to toEncode
toEncode = df_domoticz.iloc[:, 1].values
encoded = preprocessing.LabelEncoder()
encoded.fit(toEncode)
newEncode = encoded.transform(toEncode)
newEncode = newEncode.reshape(-1, 1)
OHE = preprocessing.OneHotEncoder(sparse=False)
print (OHE.fit_transform(newEncode))
toEncode = OHE.fit_transform(newEncode)

#Takes the time of alert and calculates the time between alerts in seconds

#Creates an empty column for timedelta
df_domoticz['_timedelta'] = np.nan
#Parses the time from a cell

"""cellTime = df_domoticz.iloc[0]['_id']
cellTime1 = df_domoticz.iloc[1]['_id']
cellTime = cellTime.to_datetime()
cellTime1 = cellTime1.to_datetime()
tdelta = cellTime1 - cellTime
seconds = tdelta.total_seconds()"""

#Loop to place time lapsed between alerts into _timedelta
from datetime import timedelta, datetime
numRows = df_domoticz.shape[0]
for j in range(numRows):
    currentCellTime = df_domoticz.iloc[j]['_id']
    prevCell = j-1
    prevCellTime = df_domoticz.iloc[prevCell]['_id']
    currentCellTime = currentCellTime.to_datetime()
    prevCellTime = prevCellTime.to_datetime()
    timeDelta = currentCellTime - prevCellTime
    timePassed = timeDelta.total_seconds()
    df_domoticz.iloc[j, df_domoticz.columns.get_loc('_timedelta')] = timePassed
print('Check: _timedelta')

#-----------------------------------------------------------------------------------
#Placing the KNN algorithm into practice
X = toEncode
y = df_domoticz.iloc[:, 4].values # Chosing which columns will be the variables
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #Setting the classifier to use euclidean distance as reference
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# Matrix shows the amount of correct predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# Take note of which columns you will be analyzing 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Training Test')
plt.xlabel('Time Since Last Alert')
plt.ylabel('Sensors')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Test Test')
plt.xlabel('Time Since Last Alert')
plt.ylabel('Sensors')
plt.legend()
plt.show()


