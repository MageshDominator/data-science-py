# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:26:53 2019

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


class LogRegWithRegularization(object):

    def __init__(self, n_iterations = 10000, use_optimizer = True, reg_param = 0.01, learning_rate = 0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = Regularization(alpha = reg_param)
        self.use_optimizer = use_optimizer

    def initialize_weights(self, n_features):
        # initialize the weights based on number of features
        limit = 1 / math.sqrt(n_features)
        weights = np.random.uniform(-limit, limit, (n_features,))
        return weights

    # using sigmoid function for prediction
    def sigmoid(self, X, weights):
        z = X.dot(weights)
        h = 1 / (1 + np.exp(-z))
        return h

    def loss(self, weights, X, y):
       y_pred = self.sigmoid(X, weights)
       # calculating the cost / loss of the model
       J = np.mean(-(y * np.log(y_pred)) - ((1 - y) * np.log(1 - y_pred)) + self.regularization(self.weights))
       return J

    def gradient_descent(self, weights, X, y):
        y_pred = self.sigmoid(X, weights)
        # Calculate gradient
        grad = - (y - y_pred).dot(X) + self.regularization.grad(weights)
        # return grad / X.shape[0]
        return grad

    # Train the model
    def fit(self, X, y):
        # add bias units
        X = np.insert(X, 0, 1, axis = 1)

        self.training_errors = []
        self.weights = self.initialize_weights(n_features = X.shape[1])

        # Gradient Descent using n_iterations
        if not self.use_optimizer:
            for i in range(self.n_iterations):
                # Find loss / training error
                mse = self.loss(self.weights, X, y)
                self.training_errors.append(mse)
                grad = self.gradient_descent(self.weights, X, y)
                # gradient update
                self.weights -= self.learning_rate * grad
        else:
            result = op.minimize(fun = self.loss, x0 = self.weights, args = (X, y),
                                       method = "TNC", jac = self.gradient_descent)
            self.weights = result.x

    def predict(self, X):
        X = np.insert(X, 0, 1, axis = 1)
        y_pred = self.sigmoid(X, self.weights)
        y_pred = np.round(y_pred).astype(int)
        return y_pred