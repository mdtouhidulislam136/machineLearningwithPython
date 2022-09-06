# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:02:40 2021

@author: Asus
"""

#Exercise 1

with open("auto-mpg.names.txt",'r') as data_file:
    for line in data_file:
        data = line.split()
        print(data)
        
#Exercise 2

import pandas as pd
     
df = pd.read_csv('auto-mpg.data-original.txt', header=None, names=['mpg', 'sylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name'])

print(df.head())

#Exercise 3
import pandas as pd
     
df = pd.read_csv('auto-mpg.data-original.txt', header=None, names=['mpg', 'sylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name'])

print(df.info())
print(df.max())
print(df.describe())
print(df.min())

#Exercise 4
import pandas as pd
     
df = pd.read_csv('auto-mpg.data-original.txt', header=None, names=['mpg', 'sylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name'])

print(df.isna())
# there was missing data in mpg
#--------------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv('auto-mpg.data-original.txt', header=None, names=['mpg', 'sylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name'])

print(df.any())



#Exercise 5
import pandas as pd
import seaborn as sns
df = pd.read_csv('auto-mpg.data-original.txt', header=None, names=['mpg', 'sylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'model_year', 'origin', 'car_name'])

sns.pairplot(df)
#Exercise 6

        
        
         