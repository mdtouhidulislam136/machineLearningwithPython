# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:24:28 2021

@author: Asus
"""

Ex-1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y_line = 0.5*x + 1

y_data = y_line + np.random.randn(100,)/2

df = pd.DataFrame({'x': x, 'y': y_data})

sns.scatterplot(x='x', y='y', data=df)

a, b = np.polyfit(x, y_data, deg=1)
print(a, b)

fig, ax = plt.subplots()
sns.lmplot(x='x', y='y', data=df)
plt.plot(x, y_line, 'r')



Ex-2
import pandas as pd

with open('auto-mpg.names.txt', 'r') as f:
    names = f.read().split('\n')
    
    
read_data = pd.read_csv('auto-mpg.data-original.txt', header=None, sep='\s+', quotechar="\"", skipinitialspace=True)
read_data.columns = names
print(read_data)
    
Ex-3

import pandas as pd

with open('auto-mpg.names.txt', 'r') as f:
    names = f.read().split('\n')

read_data = pd.read_csv('auto-mpg.data-original.txt', header=None, sep='\s+', quotechar="\"", skipinitialspace=True)
read_data.columns = names
read_data.dropna(inplace=True)
print(read_data)
    
Ex-4
import pandas as pd
import sklearn.model_selection as model_selection
with open('auto-mpg.names.txt', 'r') as f:
    names = f.read().split('\n')
    
    
read_data = pd.read_csv('auto-mpg.data-original.txt', header=None, sep='\s+', quotechar="\"", skipinitialspace=True)
read_data.columns = names

dataset = read_data.drop(['acceleration', 'model year', 'origin'],axis = 1)
dataset
    
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 100)

print(X_train,X_test, y_train, y_test)

Ex-5

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

Ex-6
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

Ex-7


