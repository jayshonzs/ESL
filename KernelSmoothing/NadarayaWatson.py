'''
Created on 2014-6-21

@author: xiajie
'''
import numpy as np
import SimulateData as sd
import matplotlib.pyplot as plt

def Epanechnikov(x0, x, lmbda):
    t = np.abs(x0-x)/lmbda
    if t > 1:
        return 0
    else:
        return 0.75*(1-t**2)

def ave(x0, X, Y, lmbda):
    molecular = 0.
    denominator = 0.
    for i in range(len(X)):
        kernel = Epanechnikov(x0, X[i], lmbda)
        molecular += kernel*Y[i]
        denominator += kernel
    return molecular/denominator

def draw(X, Y):
    lmbda = 0.2
    predictors = np.arange(0., 1., 0.002)
    responses = np.zeros(500)
    for i in range(len(predictors)):
        responses[i] = ave(predictors[i], X, Y, lmbda)
    plt.plot(predictors, responses)

if __name__ == '__main__':
    X, Y = sd.simulate(100)
    plt.plot(X, np.sin(4*X))
    plt.plot(X, Y, 'ro')
    draw(X, Y)
    plt.show()
