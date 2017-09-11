"""
EricNet.py
Eric Poppenheimer.  Based on work by Michael Nielsen

This module defines a Network class which can run the networks I have trained
"""

import numpy as np

#==============================================================================  

class Sigmoid(object):
# this represents a sigmoid activation function
    name = "Sigmoid"
    
    @staticmethod
    def __call__(x):
        return 1.0/(1.0+np.exp(-x))

    @staticmethod
    def prime(x):
        return self(x)*(1-self(x))
        
#==============================================================================  

class Softmax(object):
# this represents a softmax activation function
    name = "Softmax"
    
    @staticmethod
    def __call__(x):
        x = np.exp(x - np.max(x))
        return x / x.sum()
        
#==============================================================================  

class Tanh(object):
# this represents a tanh activation function
    name = "Tanh"
    
    @staticmethod
    def __call__(x):
        return np.tanh(x)

    @staticmethod
    def prime(x):
        return np.cosh(x)**-2
        
#==============================================================================  

class Network(object):
# the network class is the base class for this project.  I am defining my matrices to conform to the notation that output 
# = activator (weights * input + bias)
    def __init__(self, filePath=""):
    # filePath is added to the beginning of the filenames used while loading the net.  If filePath is left blank, load the
    # net from files stored in the directory the script was loaded from.
        self.cost = 'l' # q -> quadratic cost.  l -> log-likelihood
        self.load(filePath)

    def feedForward(self, a):
    # evaluates the network on input a
        for l in self.layers:
            a = self.activators[l](np.dot(self.weights[l], a) + self.biases[l])
        return a

    def evaluate(self, testData):
    # evaluates the performance of the neural network by comparing the output to the correct answer in the testData
        testResults = [(np.argmax(self.feedForward(x)), y) for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)
        
    def load(self, filePath):
    # Load the net from files
        # check that the last char is a '/'
        if filePath[-1] != '/':
            filePath += '/'
    
        structure = np.load(filePath + "structure.npy")
        
        # Setup the layers.        
        # self.layers is just a convenient iterator for iterating through the layers.
        # self.sizes is the size of each layer in the network  It really isn't needed for this demonstration, but I am 
        # keeping it here for compatability with my other code
    
        self.sizes = structure[0]
        
        self.biases = np.load(filePath + "biases.npy")
        self.weights = np.load(filePath + "weights.npy")
        self.layers = range(len(self.biases))
        
        activatorDict = {"Sigmoid": Sigmoid(), "Tanh": Tanh(), "Softmax": Softmax()}
        self.activators = [activatorDict[name] for name in structure[1]]
