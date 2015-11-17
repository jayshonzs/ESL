'''
Created on 2014-8-28

@author: xiajie
'''
import numpy as np

def eigen_decomposition(X):
    cov = np.transpose(X).dot(X)
    w, v = np.linalg.eig(cov)
    return w, v

'''
uncorrelated standardization
'''
def standardizing(x, L, U, mean):
    L_I = np.linalg.inv(L)
    U_T = np.transpose(U)
    y = (L_I**0.5).dot(U_T).dot(x-mean)
    return y

if __name__ == '__main__':
    X = np.zeros()
    N = len(X)
    D = len(X[0])
    w, U = eigen_decomposition(X)
    L = np.diag(w)
    mean = np.mean(X, axis=0)
    Y = np.zeros((N, D))
    for i in range(N):
        Y[i] = standardizing(X[i], L, U, mean)
