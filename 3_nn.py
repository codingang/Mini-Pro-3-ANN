#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:12:47 2017

@author: hongboing
"""

# implement a neural network model and train it using a known data set 
# import necessary libraies
import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel  

# sigmoid and desigmoid functions
# compute sigmoid for the genetated x for each node 
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y
# conpute the derivative of the sigmoid function 
def desigmoid(x):
    return x*(1-x)

# known training dataset: employed as arrays in python for easy manipulation and plotting 
# input matrix: contains 6 inputs of x and 160 records
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

# output matrix: 160 rows of y 
Y = np.array([[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
               0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]]).T
    
# known testing dataset: 4 inputs of x and 24 records
testX = np.array([[0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,1,1,1],
              [0,0,1,1,1,1],[0,1,1,0,0,0],[1,0,1,1,1,1],[1,1,1,0,0,0]])

# output matrix: 24 rows of y     
testY = np.array([[0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1]]).T

# define alpha and hidden layer size 
a = 0.001
hiddenLayer = 100

# generate random seed for initializing the weights 
np.random.seed(1)

# initialize w0 and w1 for X and Y 
# weights should be some random number between -1 and 1
w0 = 2*np.random.random((6,hiddenLayer)) - 1
w1 = 2*np.random.random((hiddenLayer,1)) - 1

# simple neutral network with 1 hidden layer 
# for 50000 runs
# l0 - input layer
# l1 - hidden layer 
# l2 - output layer 
for i in range(50000):
    # pass X as layer0 
    l0 = X
    # calculate and pass the new inputs to next layer 
    # tansfer the new inputs by using sigmoid function
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))
    # calcualte the error between real Y and calculated Y 
    l2_err = l2 - Y
    
    # feed forward
    # calculate the mean accuracy of this nn
    # print the result to show that the mean accuracy increase after each epoch 
    if (i % 10000) == 0:
        print("Mean Accuracy: "+str(1-np.mean(np.abs(l2_err))))
    
    # backpropagation: update weights back from l2 
    # use de-sigmoid function
    l2_w = l2_err*desigmoid(l2)
    l1_err = l2_w.dot(w1.T)
    l1_w = l1_err*desigmoid(l1)
    
    # update weight after backpropagation
    w1 = w1 - a * (l1.T.dot(l2_w))
    w0 = w0 - a * (l0.T.dot(l1_w))
    
# result on known training dataset 
print("\nResult on Traing dataset X and Y: ")

# l2 - training set predict y corresponsding to Y 
# l2 need to be transfered for plotting later 
print(l2.T)
print("\n")

# initialize w3 and w4 for testing dataset
# weights should be some random number between -1 and 1
w3 = 2*np.random.random((6,hiddenLayer)) - 1
w4 = 2*np.random.random((hiddenLayer,1)) - 1

# test model on testing dataset 
# l3 - input layer
# l4 - hidden layer 
# l5 - output layer 
for i in range(50000):
    # pass testX as layer0 
    l3 = testX
    # calculate and pass the new inputs to next layer 
    # tansfer the new inputs by using sigmoid function
    l4 = sigmoid(np.dot(l3,w3))
    l5 = sigmoid(np.dot(l4,w4))
    # calcualte the error between real Y and calculated Y 
    l5_err = l5 - testY
 
    # feed forward
    # calculate the mean accuracy of this nn
    # print the result to show that the mean accuracy increase after each epoch 
    if (i % 10000) == 0:
        print("Mean Accuracy: " + str(1-np.mean(np.abs(l5_err))))
    
    # backpropagation: update weights back from l2 
    # use de-sigmoid function
    l5_w = l5_err*desigmoid(l5)
    l4_err = l5_w.dot(w1.T)
    l4_w = l4_err*desigmoid(l4)
    
    # update weight after backpropagation   
    w4 = w4 - a * (l4.T.dot(l5_w))
    w3 = w3 - a * (l3.T.dot(l4_w))

# result on known testing dataset
print("\nResult on Testing dataset testX and testY: ")

# l5 - testing set predict y corresponsding to testY 
print(l5.T)
print("\n")

# create the index for predict y and real y for plotting 
size = testY.shape[0]
indexArr = np.zeros(size)
for i in range(size):
    indexArr[i] = i
 
# scatter plot: l5 v.s. testY
scatter(indexArr, testY, marker='o', c='b')  
scatter(indexArr, l5, marker='x', c='r')  
xlabel('index')  
ylabel('Real Y and Predictions of Testing Dataset')  
legend(['Real Y', 'Predict Y'])  
show()
