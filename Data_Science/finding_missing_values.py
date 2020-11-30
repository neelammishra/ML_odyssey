# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:18:51 2020

@author: Neelam
"""

import pandas as pd
import numpy as np

# creating a data 
df1 = {'Subject':['Maths','Physics','Chemistry','Computer Science',
                  'English','Physical Education'],
       'Score':[67,55,np.nan,74,np.nan,90]}
df1 = pd.DataFrame(df1,columns =['Subject','Score'] )
print(df1)

#To find out whether there is 
#any missing value
df1.isnull()

#to find missing values across columns
df1.isnull().any()

# no.of missing values in each column
df1.isnull().sum()

