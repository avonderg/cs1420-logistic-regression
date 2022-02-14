#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
from re import L
import numpy as np
from sqlalchemy import false


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03  # DO NOT TUNE THIS PARAMETER
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        # TODO
        converge = False
        epoch = 0
        while (not converge):
            epoch+=1
            # shuffle examples AND labels -> so the labels are the same (both the same way)
            shuffler = np.random.permutation(len(X))
            X_shuffled = X[shuffler]
            Y_shuffled = Y[shuffler]
            for i in range(0, len(X)/self.batch_size):
                xBatch = X_shuffled[i*self.batch_size: (i+1)*self.batch_size]
                yBatch = Y_shuffled[i*self.batch_size: (i+1)*self.batch_size]
                L = np.zeros(self.batch_size)
                for x,y in zip(xBatch, yBatch):
                    for j in range(0, self.n_classes):
                        # gradient = np.gradient(L, self.weights, axis=0)
                        if (y == j):
                            L += np.matmul(softmax(np.matmul(x, self.weights[j]) - 1), x)
                        else:
                            L += np.matmul(softmax(np.matmul(x, self.weights[j])), x)
                self.weights = (self.weights - ((np.matmul(self.alpha, L)) / len(X)))
            val = self.loss(self, xBatch, yBatch)[epoch] - self.loss(self, xBatch, yBatch)[epoch-1]
            if (val < self.conv_threshold):
                converge = True
                break

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        loss=0
        predictions = self.predict(X)
        for i in range(0, len(X)):
            loss += loss(predictions[i], Y[i])
        loss = loss/len(X)
        return loss

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        weights = self.weights
        return np.matmul(X, weights)


    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        pass
