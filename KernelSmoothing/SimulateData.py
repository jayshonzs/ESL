'''
Created on 2014-6-19

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt

def simulate(n=100):
    X = np.random.random(n)
    X = np.sort(X)
    Y = np.zeros(n)
    for i in range(n):
        e = np.random.normal(0., 0.33)
        Y[i] = np.sin(4*X[i])+e
    return X, Y

def draw(X, Y):
    plt.plot(X, np.sin(4*X))
    plt.plot(X, Y, 'ro')
    plt.show()

if __name__ == '__main__':
    X, Y = simulate()
    draw(X, Y)
