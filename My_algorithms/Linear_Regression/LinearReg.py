# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:21:40 2019

@author: MAGESHWARAN
"""

import numpy as np
import math
from scipy import optimize as op

class Regularization(object):

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return 0.5 * self.alpha * weights.T.dot(weights)

    def grad(self, weights):
        return self.alpha * weights.T.dot(weights)


class LinearRegWithRegularization(object):

    # constructor to initialize params while creating object
    def __init__(self, n_iterations = 10000, learning_rate = 0.001, reg_param = 0, use_optimizer = True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = Regularization(alpha = reg_param)
        self.use_optimizer = use_optimizer

    def initialize_weights(self, n_features):
        # initialize weights randomly
        limit = 1 / math.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features,))

    def loss(self,weights, X, y):
        # predict y with recently updated weights
        y_pred = X.dot(weights)
        # Calculate mean squarred error
        mse = 0.5 * np.mean((y - y_pred) ** 2 + self.regularization(weights))
        # print(mse)
        return mse

    def gradient_descent(self,weights, X, y):
        # predict y with recently updated weights
        y_pred = X.dot(weights)
        # find the gradient
        grad = - (y - y_pred).dot(X) + self.regularization.grad(weights)
        # return grad / X.shape[0]
        return grad

    # training the model
    def fit(self, X, y):
        # adding bias terms: X0
        X = np.insert(X, 0, 1, axis = 1)
        self.training_errors = []
        self.initialize_weights(n_features = X.shape[1])
        if not self.use_optimizer:
            # Perform Gradient descent with n_iteraions
            for i in range(self.n_iterations):
                mse = self.loss(self.weights, X, y)
                self.training_errors.append(mse)
                # gradient of loss
                grad = self.gradient_descent(self.weights, X, y)
                # update weights
                self.weights += self.learning_rate * grad
        else:
            # use optimizer from scipy
            result = op.minimize(fun = self.loss, x0 = self.weights,
                                 args = (X, y), method = "TNC",
                                 jac = self.gradient_descent)
            self.weights = result.x

    def predict(self, X):
        X = np.insert(X, 0, 1, axis =1)
        y_pred = X.dot(self.weights)
        return y_pred