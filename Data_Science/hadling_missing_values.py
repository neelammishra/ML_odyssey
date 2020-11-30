# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:47:56 2020

@author: Neelam
"""
import numpy as np
import pandas as pd

# create data
df = {'Name':['Neelam','Debasis','Monu','Monu','Resma','Niki','Urvasi','Arna',np.nan],
       'Place':['BBSR','BBSR','Rourkela','Rourkela','sundergarh',np.nan,'Bargarh','Boudh',np.nan],
       'Gender':["F","M","F","F","F","F","F",np.nan,np.nan],
       'Score':[63,48,56,56,75,np.nan,77,np.nan,np.nan]}

df1 = pd.DataFrame(df,columns = ['Name','Place',
                                  'Gender','Score'])
print(df1)

## Looking for missing values
df1.isnull().any()
## No.of missing values in each column
df1.isnull().sum()

## drop all the values having null values = losing
# a lot of values
df1.dropna()

## drop only if entire row has missing value
df1.dropna(how = 'all')

## drop only if a row has more thab 2 NaN values
df1.dropna(thresh = 2)

## drop nan in a specific column
df1.dropna(subset=['Gender'])

## replacing missing values with 0
df1.fillna(0)

## replacing missing value with mean
df1['Score'].fillna(df1['Score'].mean(),
inplace=True)
print(df1)

## replacing missing value with mean
df1['Score'].fillna(df1['Score'].median(),
inplace=True)
print(df1)

#replacing  particular value 
#with another value
print(df1.replace({'F':'U'}))



print(df1)











