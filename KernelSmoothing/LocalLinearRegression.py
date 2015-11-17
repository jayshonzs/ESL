'''
Created on 2014-6-21

@author: xiajie
'''
import numpy as np
import SimulateData as sd
import matplotlib.pyplot as plt

def augment(X):
    lst = [[1, X[i]] for i in range(len(X))]
    return np.array(lst)

def Epanechnikov(x0, x, lmbda):
    t = np.abs(x0-x)/lmbda
    if t > 1:
        return 0
    else:
        return 0.75*(1-t**2)

def kernel(x0, x, lmbda):
    return Epanechnikov(x0, x, lmbda)

def Weight(x0, X, lmbda):
    W = np.zeros(len(X))
    for i in range(len(X)):
        W[i] = kernel(x0, X[i], lmbda)
    return np.diag(W)

def draw(X, Y):
    lmbda = 0.2
    predictors = np.arange(0., 1., 0.005)
    responses = np.zeros(200)
    B = augment(X)
    BT = np.transpose(B)
    for i in range(len(predictors)):
        b = np.zeros(2)
        b[0] = 1
        b[1] = predictors[i]
        W = Weight(predictors[i], X, lmbda)
        inv = np.linalg.inv(BT.dot(W).dot(B))
        responses[i] = np.transpose(b).dot(inv).dot(BT).dot(W).dot(Y)
    plt.plot(predictors, responses)
    
if __name__ == '__main__':
    X, Y = sd.simulate(100)
    plt.plot(X, np.sin(4*X))
    plt.plot(X, Y, 'ro')
    draw(X, Y)
    plt.show()
