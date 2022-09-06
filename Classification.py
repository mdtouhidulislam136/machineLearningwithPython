# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:27:27 2021

@author: Asus
"""

Exercise 1:
     
import pandas as pd
 
#loading data
titanic = pd.read_csv('titanic.csv')

print(titanic) 

Exercise 2:
import pandas as pd
 
titanic = pd.read_csv('titanic.csv')
titanic.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print("Explore the data: ")
print(titanic.groupby("Fare").mean())

Exercise 3:
import pandas as pd
import sklearn.model_selection as criteria_selection
titanic = pd.read_csv('titanic.csv')
titanic.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

y = titanic.Fare
x = titanic.drop('Fare', axis=1)
x_train,x_test,y_train,y_test=criteria_selection.train_test_split(x,y,test_size=0.2)

print("shape of original dataset :", titanic)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)


Exercise 4:
import pandas as pd
import sklearn.model_selection as criteria_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



titanic = pd.read_csv('titanic.csv')
titanic.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

y = titanic.Pclass
x = titanic.Fare

logisticRegr = LogisticRegression(solver='liblinear')
x_train,x_test,y_train,y_test=criteria_selection.train_test_split(x,y,test_size=0.2)
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Seaborn method
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

print("Accuracy: %.3f (%.3f)" % (score.mean(), score.std()))


    
Exercise 5:
like I am five

precision:
Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
His prediction I am like 5. Here we got prediction0.97, precision because he told like I am five.
 If he say I am five that mean 100 precision positive observations accurate.
 
recall:
Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.
We got almost 0.97 his age from prediction. That is good enough.
f-score:

F-score is average of Recall and precision
F1 Score = 2*(Recall * Precision) / (Recall + Precision)

It just calculated how good was his prediction.


Exercise 6:
    
import pandas as pd
import sklearn.model_selection as criteria_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



titanic = pd.read_csv('titanic.csv')
titanic.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

y = titanic.Survived
x = titanic.Sex

logisticRegr = LogisticRegression( penalty='none')
x_train,x_test,y_train,y_test=criteria_selection.train_test_split(x,y,test_size=0.2)
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("Accuracy: %.3f (%.3f)" % (score.mean(), score.std()))

cm = metrics.confusion_matrix(y_test)
print(cm)




    
    
    