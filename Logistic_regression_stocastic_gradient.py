'''
@Team 4: Manasa Kashyap Harinath
	 Sravanthi Avasarala
	 Pavitra Shivanand Hiremath
	 Ankit Bisht
'''
'''The aim of this implementation is to demonstrate the working of the logistic regression given an IRIS dataset. 
Essentially, we try to predict which class a plant belong, given the information about its sepal length and sepal width.
 This is achieved by using logistic regression classifier. To increase the performance, we use KFold cross validation'''

import pandas as pd
from csv import reader
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from numpy import where
import seaborn as sn


'''Dataset information: Iris dataset from the sklearn library is used. Originally, there are 4 predictor variables, out of which for the ease of implementation, we have considered the first two predictors, X (1. sepal length in cm 2. sepal width in cm). Y being the response varibale which represents the class of the plant. We consider only Setosa (0) and Versicolour(1), since logistic regression is a binary classifier.'''

iris = datasets.load_iris()
X = iris.data[:, :2] 
Y = iris.target


def FormatY(Y):

	for j in range(len(Y)):
		if(Y[j]==2):
			if(j%2==0):
				Y[j]=1
			else:
				Y[j]=0
		else:
			continue
	return Y

'''The predictor data X is split into 2 arrays, X1 and X2 respectively.'''
def splitTheData(X):
	X1=[]
	X2=[]

	for i in range(len(X)):
		X1.append(X[i][0]) # sepal length
		X2.append(X[i][1]) # sepal width
	
	return X1,X2
		

'''We calculate z by using the formula, z= b0 + b1*x1 + b2*x2. Where b0, b1 and b2 are the regression coefficients. x1 and x2 are the individual data instance. Sepal length and sepal width. After this, the prediction is made using the model, sigmoid(z). The sigmoid function is calculated using  pred = 1 / (1 + e^(-z)). Thus predicted value will be in the range of (0,1). This represents the probabilty'''
def applySigmoid(b0,b1,b2,x1,x2):
	
	z= b0 + b1*x1 + b2*x2
	pred=float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return pred,z


'''We calculate b0,b1,b2 from the previous prediction that is obtained from the sigmoid function. Here x1 and x2 are the individual data instances. y is the individual reponse class. (y-pred) represents the error in the prediction. Alpha is the learning rate. We thus return the new coefficients based on previous predictions and coefficients.'''
def calculateCoEfficients(pred, y, x1,x2,b0,b1,b2):

	alpha=0.3

	b0=b0 + alpha * ( y - pred ) * pred * ( 1 - pred ) * 1
	b1=b1 + alpha * ( y - pred ) * pred * ( 1 - pred ) * x1
	b2=b2 + alpha * ( y - pred ) * pred * ( 1 - pred ) * x2

	return b0,b1,b2
	
'''This method represents the our implementation of logistic regression. We can apply stochastic gradient descent to 
the problem of finding the coefficients for the logistic regression model as follows:

Given each training instance:

Calculate a prediction using the current values of the coefficients. (Using sigmoid function. Please refer applySigmoid method)
Calculate new coefficient values based on the error in the prediction.

The process is repeated until the model is accurate enough (e.g. error drops to some desirable level) or for a fixed number 
iterations. You continue to update the model for training instances and correcting errors until the model is accurate enough. 
Here, the process is repeated for 100 times. Each time new coefficients are calculated. The co efficients obained in the last 
iteration is considered to be the best coefficients. These coefficients (Model) is used to test against the test data and 
compute the final accuracy.

 '''

def applyHypothesis(X1,X2,Y,noOfIters):
	
	b0=0
	b1=0
	b2=0
	pred=0
	finalPred=[]
	for i in range(noOfIters):
		for j in range(len(X1)):
			x1=X1[j]
			x2=X2[j]
			if (i==0):
				pred=0.5
			else:		
				pred,z=applySigmoid(b0,b1,b2,x1,x2)	
			
			b0,b1,b2= calculateCoEfficients(pred,Y[j],x1,x2,b0,b1,b2)
			finalPred.append(pred)
	
	return b0,b1,b2

'''Our model is trained using train_set and test against vaidation set (This is obtained by the calculatekfoldSplit method). 
The model which has highest acuracy is considered for the final testing. This method returns the best model along
 with the testing set'''

def computeLogisticRegUsingKFold(X,Y,kfl):
	noOfIters=100
	accuracyArr=[]
	b0Arr=[]
	b1Arr=[]
	b2Arr=[]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state = 42)  
	for train,test in kfl.split(X_train):
		Xtest_1, Ytest_1, Xtrain_1, Ytrain_1=calculatekfoldSplit(train, test)
		X1,X2=splitTheData(Xtrain_1)
		b0,b1,b2=applyHypothesis(X1,X2,Ytrain_1,noOfIters)
		b0Arr.append(b0)
		b1Arr.append(b1)
		b2Arr.append(b2)
		X_1,X_2=splitTheData(Xtest_1)
		predfin,zarr = SigmoidForeachData(X_1,X_2,b0,b1,b2)
		predfin=FindBinaryPredictions(predfin)
		accuracy=ComputeAccuracy(Ytest_1,predfin)
		accuracyArr.append(accuracy)
		print 'Accuracy of ',len(accuracyArr),'Fold is ',accuracy
	
	b0= b0Arr[accuracyArr.index(np.amax(accuracyArr))]
	b1= b1Arr[accuracyArr.index(np.amax(accuracyArr))]  
	b2= b2Arr[accuracyArr.index(np.amax(accuracyArr))]
	return b0,b1,b2,X_test,y_test

'''The train and test obtained from KFold.split() method are the indices. In this method, we retrieve the corresponding data from X_train 
and y_train matrices. Thus, obtained Xtest_1, Ytest_1, Xtrain_1, Ytrain_1 are returned.'''
def calculatekfoldSplit(test, train):
	Xtest_1 = []
	Ytest_1 = []
	Xtrain_1=[]
	Ytrain_1 = []
	for i in range(len(test)):
		Xtest_1.append( X[test[i]])
		Ytest_1.append( Y[test[i]])
	
	for i in range(len(train)):
		Xtrain_1.append( X[train[i]])
		Ytrain_1.append( Y[train[i]])
	
	return Xtest_1, Ytest_1, Xtrain_1, Ytrain_1


'''After calculating the best model, the model has to be tested against the testing data. x1 and x2 are the individual instances'''
def SigmoidForeachData(X_1,X_2,b0,b1,b2):
	predfin=[]	
	zarr=[]
	for i in range(len(X_1)):	

		pred,z=applySigmoid(b0,b1,b2,X_1[i],X_2[i])
		predfin.append(pred)
		zarr.append(z)

	return predfin,zarr

'''The sigmoid function returns the predictions whose values ranges from 0 to 1. 
To correctly classify, we consider the predictions whose value is greater than or equal to 0.5, is considered as class 1 and lesser than 0.5 is 
considered as class 0. Hence the final predictions will be in terms of 0 or 1 '''
def FindBinaryPredictions(predfin):

	for j in range(len(predfin)):
		if(predfin[j] >= 0.5):
			predfin[j]=1
		else:
			predfin[j]=0
	return predfin

'''Confusion matrix is computed using sklearn library. Using confusiion matrix, accuracy is calculated'''
def ComputeAccuracy(y_test,predfin):
	
	(tn, fp, fn, tp) = confusion_matrix(y_test,predfin).ravel()
	
	print tn,fp,fn,tp	
	accuracy= float(float(tp+tn) / float(tn+fp+fn+tp))
	return accuracy


'''Logistic Regression is implemented without KFold. This is just for the comparison'''	
def LogisticRegressionWithoutUsingKfold(X,Y):
	
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state = 42) 
	noOfIters=100
	X1,X2=splitTheData(X_train)
	b0,b1,b2=applyHypothesis(X1,X2,y_train,noOfIters)
	X_1,X_2=splitTheData(X_test)
	predfin,zarr = SigmoidForeachData(X_1,X_2,b0,b1,b2)
	predfin=FindBinaryPredictions(predfin)
	accuracy=ComputeAccuracy(y_test,predfin)
	print 'Accuracy without KFold', accuracy

'''Logistic Regression using kFold cross validation. Here, we use 5 folds'''
def LogisticRegressionUsingKfold(X,Y):	

	kfl =KFold(n_splits=5, random_state=42, shuffle=True)
	b0,b1,b2,X_test,y_test=computeLogisticRegUsingKFold(X,Y,kfl)
	X_1,X_2=splitTheData(X_test)
	predfin,zarr = SigmoidForeachData(X_1,X_2,b0,b1,b2)
	predfin=FindBinaryPredictions(predfin)
	accuracy=ComputeAccuracy(y_test,predfin)
	
	print 'Accuracy with KFold', accuracy
	return predfin,X_test,y_test


'''Graph is plotted for to test our implementation on the randomly generated dataset. 
Here X1 and X2 (Two attributed) are plotted in x and y axes respectively.'''
def plotTheGraph(predfin,X_test,y_test):
	
	pos = where(pd.DataFrame(predfin) == 1)
	neg = where(pd.DataFrame(predfin) == 0)
	plt.scatter(X_test[pos, 0], X_test[pos, 1], marker='o', c='b',label = '1')
	plt.scatter(X_test[neg, 0], X_test[neg, 1], marker='x', c='r',label = '0')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.title('Logistic Regression on Random Dataset')
	plt.legend()
	plt.show()

Y=FormatY(Y)

'''Logistic Regression using KFold cross validation'''
predfin,X_test,y_test=LogisticRegressionUsingKfold(X,Y)
sn.heatmap(confusion_matrix(y_test,predfin), annot=True)
plt.title('Confusion Matrix')
plt.show()
'''Logistic Regression using Just test train split'''
LogisticRegressionWithoutUsingKfold(X,Y)



'''This is to demonstrate the working of the implementation on random X and random Y'''
rng = np.random.RandomState(0)
X = rng.randn(300, 2) # X is generated randomly. It is a 300x2 matrix. Two columns represent X1 and X2 (2 attributes)
Y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)  # Y is either 0 or 1
print '\n*****The results below are only for the random dataset generation*****\n'

'''The randomly generated X and Y are then given to our implementation and accuracy is computed'''
predfin,X_test,y_test=LogisticRegressionUsingKfold(X,Y) # predfin contains the array of predicted classes.

plotTheGraph(predfin,X_test,y_test) # The graph is plotted.


'''
Accuracy with Kfold = 88.88 %

'''





 
