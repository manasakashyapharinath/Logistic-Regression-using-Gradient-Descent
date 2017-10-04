# Logistic-Regression-using-Gradient-Descent
**Implementation of Logistic Regression using Stochastic Gradient Descent method**

***A brief walk through on the implementation is provided via a link below:***

https://docs.google.com/presentation/d/1WxPfVO4q7ILGcDqwTrtRc4r1tKACQKdylYZ--31WdSw/edit?ts=59d3d384#slide=id.g26db42bbd0_0_7

Also, you can find the detailed explanation through the comments above each method in the Logistic_regression_stocastic_gradient.py file.

**Dataset:** 

For our implementation we have used IRIS dataset from the sklearn library. There are 4 predictor attributes and one response attribute.
We have considered only 2 predictor attributes. Sepal_length and sepal_width and try to predict which class of plant it belongs to,
Setosa (0) or Versicolour(1).

**Implementation:**
File:  Logistic_regression_stocastic_gradient.py

We use logistic regression to predict and classify the plant into Setosa or Versicolour. For this dataset, 
the logistic regression has three coefficients just like linear regression:
                                            
                                              Prediction= 1/(1+e^ (-z) )   **Sigmoid**
                                               z= b0 + b1 * x1 + b2 * x2
                                               
                                              where b0, b1 and b2 are the co efficients.
                                              This is how we calculate b0,b1 and b2,
                                        
                                         b = b + alpha * (y – prediction) * prediction * (1 – prediction) * x
                                         
                                          using this we calculate, b0, b1 and b2 for our implementation.
                       
                                        
We can apply stochastic gradient descent to the problem of finding the above coefficients for the logistic regression model as follows:

Given each training instance:

1)Calculate a prediction using the current values of the coefficients.

2)Calculate new coefficient values based on the error in the prediction.

The process is repeated until the model is accurate enough (e.g. error drops to some desirable level) or for a fixed number iterations.
For our implementation, the gradient descent is repeated 100 times to find the best accuracy.

**Kfold Validation:**

We split the dataset into trainin and testing set. We then apply the KFold cross validation by splitting the training data
into training and validation set. Since we have used K=5, our logistic model is trainied accross all the fold and the model which has best accuracy is then test against the testing set and final accuracy is recorded. Our accuracy is found to be 88.88% with Kfold cross validation.

**Sklearn Implementation:**

File: logistic_using_sklearn.py

We have implemented the LogisticRegression() from sklearn library using kfold validation and computed the accuracy and it was found to be 80%.

**Random Data:**

The implementation is tested against random data and respective graphical representation is shown. One can find this in
Logistic_regression_stocastic_gradient.py file. The accuracy for the random generated data is 64.44%



