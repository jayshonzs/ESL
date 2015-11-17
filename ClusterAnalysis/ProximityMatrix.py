'''
Created on 2014-8-12

@author: xiajie
'''
import numpy as np

def create_matrix(X, dissim):
    N = len(X)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = dissim(X[i], X[j])
    return mat

def squared_dist(x1, x2):
    return np.sum((x1-x2)**2)

'''
for quantitative variables
'''
def sqrd_dissim_mat(X):
    return create_matrix(X, squared_dist)

def ro(x1, x2):
    P = len(x1)
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    numerator = 0.
    denominator1 = 0.
    denominator2 = 0.
    for p in range(P):
        numerator  += (x1[p]-mean1)*(x2[p]-mean2)
        denominator1 += (x1[p]-mean1)**2
        denominator2 += (x2[p]-mean2)**2
    return numerator/np.sqrt(denominator1*denominator2)

'''
for quantitative variables
'''
def correlation_sim_mat(X):
    return create_matrix(X, ro)

'''
translate ordinal variables to quantitative variables
indexes={p0:M0, p1:M1}
'''
def cook_ordinal(X, indexes={}):
    for key in indexes.keys():
        p = key
        M = indexes[key]
        for i in range(len(X)):
            X[i, p] = (X[i, p]-0.5)/M
    return X

if __name__ == '__main__':
    X = np.array([[1,1,1],
                  [1,2,3],
                  [2,1,1]])
    print sqrd_dissim_mat(X)
