# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 19:09:48 2019

@author: MAGESHWARAN
"""

import math
import operator

class KNN():

    def __init__(self, k = 5):
        # number of neighbors
        self.k = k

    def euclideanDistance(self, data1, data2):
        length = len(data2) - 1
        distance = 0
        # iterate over the list data1 and data2 leaving it's output(Y)
        for i in range(length):
            distance += pow((data1[i] - data2[i]), 2)
        # returns the eulicidean distance for each test and training sample
        return math.sqrt(distance)

    def getNeighbors(self, X_test, X_train):
        distance = []
        neighbors = []
        # Single test data is given and this loops iterates over all training examples to find distance
        for i in range(len(X_train)):
            dist = self.euclideanDistance(X_test, X_train[i])
            distance.append((X_train[i], dist))
        distance.sort(key = operator.itemgetter(1))
        # choose the k_nearest examples as neighbors
        for j in range(self.k):
            neighbors.append(distance[j][0])
        return neighbors

    def getResponse(self, neighbor):
        # after grabbing the nearest neighbors, this helps us to pick the most repeated class
        classItBelongs = {}
        for i in range(len(neighbor)):
            response = neighbor[i][-1]
            if response in classItBelongs:
                classItBelongs[response] += 1
            else:
                classItBelongs[response] = 1
        # sort them on the basis of number of counts(values) in descending
        predicted = sorted(classItBelongs.items(), key = operator.itemgetter(1), reverse = True)
        # return the class type(most repeated)
        return predicted[0][0]

    def predict(self, X_test, X_train, y_train):
        # converting nd array to list
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        # append all the respective training features and outputs
        for l in range(len(X_train)):
            X_train[l].append(y_train[l])
        output = []
        # iterate over the test set and find the classtype for each
        for i in range(len(X_test)):
            # find the k number of neighbor
            neighbor = self.getNeighbors(X_test[i], X_train)
            # mark the class with more counts in k_nearest_neighbors
            result = self.getResponse(neighbor)
            output.append(result)
        return output