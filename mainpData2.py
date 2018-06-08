# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:57:39 2018

@author: Andre
"""

import json
import csv

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
