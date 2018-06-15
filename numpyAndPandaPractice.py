# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 19:09:37 2018

@author: maxjf, using code from Towards Data Science Blog by Adi Bronshtein
https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673
and other linked pages
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import datasets
data = datasets.load_boston()


"""pd.read_filetype() - 'filetype' is replaced by the type of file you
want to read
ex. pd.read_csv

can make new DataFrame objects using:
    pd.DataFrame()
    
can convert DataFrames to different kinds of files, using
df.to_filetype(filename)"""

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
dfBoston = pd.DataFrame(data.data, columns = data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

#Using statsmodels
X = dfBoston["RM"]
y = target["MEDV"]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())


"""Can get the first n rows with df.head(n), or the last n rows with df.tail(n)
df.shape gives the numbers of rows and columns
df.info() gives you the index, datatype, and memory information
s.value_counts(dropna=False) would allows you to view unique values and counts
for a series(like a column or a few column)

df.describe() inputs summary statistics for numerical columns

df.mean() returns the mean of all columns
df.corr() returns the correlation between columns in a data frame
df.count() returns the number of non-null values in each data frame column
df.max() returns the highest value in a column
df.min() returns the lowest value in each column
df.median() returns the median of each column
df.std() returns the standard deviation of each column
"""