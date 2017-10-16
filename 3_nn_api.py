#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:11:04 2017

@author: hongboing
"""

# Implement a neural network model and train it using a known data set 
# import necessary libraies
import numpy as np
# library for cross validation and train/test method
from sklearn.cross_validation import train_test_split
# library for neural network and performance
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.array([[0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,1,1,0,0,0],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,1,1,0,0,0],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1]])

Y = np.array([[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]]).T

# train/test api - sepetate the dataset into a training set and a test set 
# convention: train:test = 8:2 or 6:4
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# MLP parameters: check the link below for the meanings of hyperparameters 
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 

# note: we could also try adjust the hidden_layer_sizes and momentum 
# extra layers may cause overfitting 

# Modify hyperparameters trails 
# Modifications inlcude: solver, hidden layer size, and actication function

clf_self = MLPClassifier(solver='sgd',alpha=0.001, hidden_layer_sizes=100,activation = 'logistic')

# API default hyperparameters 
# API default: solver adam, hidden layer = 100, activation = relu
# relu: the rectified linear unit function, returns f(x) = max(0, x)
# hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
clf15 = MLPClassifier()

# models fit on train and test datasets
clf_self.fit(X_train,Y_train)
clf_self.fit(X_test,Y_test)

# API default model fit and predict - best model selected 
clf15.fit(X_train,Y_train)
Y_pred1 = clf15.predict(X_test)
Y_pred2 = clf_self.predict(X_test)

# print out the results
print('\n')
print('Self implemented model hyperparameter: solver = sdg, alpha=0.001, hidden_layer = 100, activation = logistic')
print('\n')
print('Api model performance with same hyperparamer set as self-implemented model: ')
print('Mean accuracy on train: ', clf_self.score(X_train,Y_train))
print('Mean accuracy on test: ', clf_self.score(X_test,Y_test))
print('Accuracy score on test Y and predict Y: ',
      accuracy_score(Y_test, Y_pred2, normalize=True, sample_weight=None))

print('\n')
print('API default model hyperparameter: solver = adam, hidden_layer = 100, activation = relu')
print('\n')
print('API default model performance: ')
print('Mean accuracy on train: ', clf15.score(X_train,Y_train))
print('Mean accuracy on test: ', clf15.score(X_test,Y_test))
print('Accuracy score on test Y and predict Y: ',
      accuracy_score(Y_test, Y_pred1, normalize=True, sample_weight=None))
print('\n')
