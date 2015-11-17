'''
Created on 2014-8-28

@author: xiajie
'''
import numpy as np

def eigen_decomposition(X, M=2):
    cov = np.transpose(X).dot(X)
    w, v = np.linalg.eig(cov)
    indexes = {}
    for i in range(len(w)):
        eigen = w[i]
        indexes[eigen] = i
    lw = w.tolist()
    lw.sort(reverse=True)
    principal_eigen_vectors = []
    for m in range(M):
        idx = indexes[lw[m]]
        principal_eigen_vectors.append(v[:, idx])
    return lw[:M], principal_eigen_vectors

def project(X):
    N = len(X)
    X2d = np.zeros((N, 2))
    eigenvectors = eigen_decomposition(X)[1]
    for i in range(N):
        x = X[i]
        X2d[i, 0] = np.transpose(x).dot(eigenvectors[0])
        X2d[i, 1] = np.transpose(x).dot(eigenvectors[1])
    return X2d

def draw(X2d):
    pass

if __name__ == '__main__':
    X = np.zeros()
    X2d = project(X)
    draw(X2d)
