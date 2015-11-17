'''
Created on 2014-8-28

@author: xiajie
'''
import numpy as np

def centering(X):
    N = len(X)
    D = len(X[0])
    centered = np.zeros((N, D))
    mean = np.mean(X, axis=0)
    for i in range(N):
        centered[i] = X[i] - mean
    return centered

def eigen_decomposition(X):
    cov = X.dot(np.transpose(X))
    w, v = np.linalg.eig(cov)
    return w, v

def true_eigens(X, w, v):
    pass

if __name__ == '__main__':
    X = np.zeros()
    X = centering(X)
    w, v = eigen_decomposition(X)
