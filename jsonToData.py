# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:15:34 2018

@author: Andre
"""

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

#take data from .json and put it in a DataFrame
data = []
with open("domoticz-2018-03-19.json") as fp:
    for line in fp:
        data.append(json.loads(line))
    df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
print(df.head(20))

#describe our dataset
print(df.shape)
print(df.dtypes)
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None)
pd.DataFrame([range(10)])
print(df.describe())