'''
Created on 2014-6-21

@author: xiajie
'''
import numpy as np
import SimulateData as sd
import matplotlib.pyplot as plt

def ave(predictor, X, Y):
    knn = X[:30].tolist()
    knny = Y[:30].tolist()
    for i in range(30, len(X)):
        distance = np.abs(X[i] - predictor)
        if distance < np.abs(knn[0] - predictor):
            del knn[0]
            del knny[0]
            knn.append(X[i])
            knny.append(Y[i])
        else:
            break
    return sum(knny)/30.

def draw(X, Y):
    predictors = np.arange(0., 1., 0.002)
    responses = np.zeros(500)
    for i in range(len(predictors)):
        responses[i] = ave(predictors[i], X, Y)
    plt.plot(predictors, responses)

if __name__ == '__main__':
    X, Y = sd.simulate(100)
    plt.plot(X, np.sin(4*X))
    plt.plot(X, Y, 'ro')
    draw(X, Y)
    plt.show()
