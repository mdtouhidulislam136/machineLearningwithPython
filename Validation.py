# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:08:47 2021

@author: Asus
"""

Exercise-1

import pandas as pd

validaton = pd.read_csv('winequality-red.csv')

print(validaton)

Exercise-2
import pandas as pd
import sklearn.model_selection as criteria_selection
validation = pd.read_csv('winequality-red.csv')
validation.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


y = validation.quality
x = validation.drop('quality', axis=1)
x_train,x_test,y_train,y_test=criteria_selection.train_test_split(x,y,test_size=0.2)

print("shape of original dataset :", validation)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)
Exercise-3




Exercise-4

Exercise-5

Exercise-6