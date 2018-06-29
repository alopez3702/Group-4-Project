# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:07:22 2018

@author: maxjf
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import model_selection

class SofwerxKnnAlgorithm(object):
    initialDataframe = []
    path1 = ''
    path2 = ''
    pathOptional = ''
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2
        domoticzDf = self.formatDomoticz(path1)
        iftttDf = self.formatIfttt(path2)
        df = domoticzDf.append(iftttDf, ignore_index=True) #append the data
        df = self.binarizeAndOrganize(df)
        self.kNNAndConfusion(df)
        
    def parseifttt(self, d, path): #method for reading ifttt files
        listing = os.listdir(path)
        for infile in listing:
            if infile.endswith(".json"):
                for line in open(infile):
                    d.append(json.loads(line))
        return d
    
    def formatDomoticz(self,path):
        data = [] #list to store data
        listing = os.listdir(path)
        for infile in listing: #for every file in this folder
            if infile.endswith(".json"): #if it's a json file
                for line in open(infile): #read every line of the file
                    data.append(json.loads(line)) #put every line into the list
        df_domoticz = pd.DataFrame.from_dict(json_normalize(data), orient='columns') #put data from list into a DataFrame
        #choose certain columns
        df_domoticz = df_domoticz[['_id','_source.message']]
        #find the size of the column and store it
        #print(df.shape)
        numRows = df_domoticz.shape[0]
        #print(numRows)
        uniqueMessageList = []
        #df.head(100)
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
        for k in reversed(range(len(to_delete))):
            df_domoticz.drop(df_domoticz.index[to_delete[k]], inplace=True)
        df_domoticz.reset_index(drop = True, inplace = True) #Reset the index for further looping
        
        df = df_domoticz
        #format data
        df._id = pd.to_datetime(df._id) #turn _id column to a datetime object
        df['intrusion'] = [0] * df.shape[0] #adds intrusion data
        df = df[['_id', 'Sensor Type', 'Sensor State', 'intrusion']] #selects rows to be used
        df['dates'] = [d.date() for d in df['_id']]
        
        m = 0
        numRows = df.shape[0]
        for k in range(numRows):
            i = df['dates'][k]
            flag = i.strftime('%Y-%m-%d') == "2018-04-18" or i.strftime('%Y-%m-%d') == "2018-04-13" or i.strftime('%Y-%m-%d') == "2018-04-12"
            if flag:
                df.loc[k, 'intrusion'] = 1
            m = m + 1
        df = df[['_id', 'Sensor Type', 'Sensor State', 'intrusion']]
        return df
    
    def formatIfttt(self,path):
        data_ifttt = [] #array for parsing ifttt data
        
        data_ifttt = self.parseifttt(data_ifttt, path)
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
        numRows = df_ifttt.shape[0]
        for k in range(numRows):
            i = df_ifttt['dates'][k]
            flag = i.strftime('%Y-%m-%d') == "2018-04-18" or i.strftime('%Y-%m-%d') == "2018-04-13"
            if flag:
                df_ifttt.loc[k, 'intrusion'] = 1
            n = n + 1
        
        df_ifttt = df_ifttt[['_id', 'Sensor Type', 'Sensor State', 'intrusion']]
        df_ifttt = df_ifttt.sort_values(by=['Sensor Type', '_id'])
    
    def binarizeAndOrganize(self, dataFrame):
        df = dataFrame[pd.notnull(dataFrame['Sensor State'])]
        df = df[pd.notnull(df['Sensor Type'])]
        
        df['hour'] = [d.hour for d in df['_id']]
        df['minute'] = [d.minute for d in df['_id']]
        df['month'] = [d.month for d in df['_id']]
        df['day'] = [d.day for d in df['_id']]
        initialDataframe = df
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
        columnsState=['1', '2', '3', '4', '5', '6']
        states = pd.DataFrame(Y, columns=columnsState)
        
        #append the binarized data to the main array (also removes 'Sensor State' and 'Sensor Type)
        df = df[['intrusion', 'month', 'day', 'hour', 'minute']]
        df = df.join(types)
        df = df.join(states)
        return df
    def kNNAndConfusion(self, dataFrame):
        X = dataFrame.iloc[:, 1:28].values
        y = dataFrame.iloc[:, 0:1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        c = classifier.fit(X_train, y_train)
        
        y_pred = c.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        result = c.score(X_test, y_test)
        seed = 7
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        scoring = 'roc_auc'
        #results = model_selection.cross_val_score(classifier, X, y, cv=kfold, scoring=scoring)
        
        def confusionMatrixString(cm):
            print("Confusion Matrix:")
            print("True Positive; False Positive; True Negative; False Negative")
            c = str(cm[0][0]) + "            " + str(cm[0][1]) + "              " + str(cm[1][1]) + "             " + str(cm[1][0])
            return c
        
        def accuracyString(r):
            r = r*100
            r = round(r, 3)
            a = "Accuracy: " + str(r)
            return a
        
        def rocString(roc):
            print("Area under ROC:")
            print("Mean;  Standard Deviation")
            mean = round(roc.mean(), 3)
            s = round(roc.std(), 3)
            c = str(mean) + "; " + str(s)
            return c
        
        print(confusionMatrixString(cm))
        print()
        print(accuracyString(result))
        print()
        #print(rocString(results))
        print()
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        
        #plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('r', 'g')))
        
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        """for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j], X_set[y_set == j], c = ListedColormap(('red', 'green'))(i), label = j)
            print("Hello world")"""
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', None)
        pd.set_option('date_yearfirst', True)
        pd.set_option('display.max_rows', 2500)

    
    def analyzeNewData(self, pathName, fileType):
        pass
    
    
path1 = 'C:/Users/maxjf/Summer Internship 2018/Final algorithm' 
path2 = 'C:/Users/maxjf/Summer Internship 2018/Final algorithm'    
practice = SofwerxKnnAlgorithm(path1, path2)    
