'''
Created on 2014-8-28

@author: xiajie
'''
import numpy as np

def eigen_decomposition(X, M=5):
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

def compress(x, eigenvectors, mean):
    D = len(x)
    M = len(eigen_vectors)
    dynamic = np.zeros(D)
    for i in range(M):
        dynamic = dynamic + np.transpose(x).dot(eigenvectors[i]).dot(eigenvectors[i])
    constant = np.zeros(D)
    for i in range(M+1, D):
        constant = constant + np.transpose(mean).dot(eigenvectors[i]).dot(eigenvectors[i])
    return dynamic + constant

if __name__ == '__main__':
    X = np.zeros()
    compressed_X = []
    mean = np.transpose(np.mean(X, axis=0))
    eigen_values, eigen_vectors = eigen_decomposition(X, M=10)
    for i in range(len(X)):
        x = X[i]
        compressed_x = compress(x, eigen_vectors, mean, M=10)
        compressed_X.append(compressed_x)
