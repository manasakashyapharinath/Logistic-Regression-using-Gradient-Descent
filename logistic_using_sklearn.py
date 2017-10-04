'''
Logistic regression using SKLearn library. We have computed confusion matrix, plot the graph to show the classification. 
Finally, accuracy is computed

Accuracy = 80%
'''

import pandas as pd
import numpy as np
import csv
import seaborn as sns
from csv import reader
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier,SGDRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sys
from sklearn import model_selection
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target


for j in range(len(Y)):
	if(Y[j]==2):
		if(j%2==0):
			Y[j]=1
		else:
			Y[j]=0
	else:
		continue


rkf = model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
for train_index,test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

model = LogisticRegression()
model.fit(X_train,y_train)
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected, predicted))
print 'Confusion matrix is given below'
print(metrics.confusion_matrix(expected, predicted))
print('The accuracy using Sklearn is ')
print(metrics.accuracy_score(expected, predicted))
plot_decision_regions(X_test,y_test, clf=model)
plt.show()
