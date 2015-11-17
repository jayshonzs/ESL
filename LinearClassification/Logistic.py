'''
Created on 2014-6-7

@author: xiajie
'''
import numpy as np
import SAHeart_data as hd

def augment(data):
    dlist = data.tolist()
    for i in range(len(dlist)):
        dlist[i].insert(0, 1.)
    return np.array(dlist)

def cp(x, beta):
    lf = x.dot(beta)
    return np.exp(lf)/(1+np.exp(lf))

def W(X, beta):
    w = np.zeros(len(X))
    for i in range(len(X)):
        pi = cp(X[i],beta)
        w[i] = pi*(1-pi)
    return np.diag(w)

def Z(X, Y, W, beta):
    P = np.zeros(len(X))
    for i in range(len(P)):
        P[i] = cp(X[i], beta)
    return X.dot(beta) + np.linalg.inv(W).dot(Y-P)

def logistic(X, Y):
    beta_old = np.zeros(len(X[0]))
    XT = np.transpose(X)
    while True:
        w = W(X, beta_old)
        z = Z(X, Y, w, beta_old)
        beta_new = np.linalg.inv(XT.dot(w).dot(X)).dot(XT).dot(w).dot(z)
        diff = np.sum((beta_new-beta_old)**2)
        if np.sqrt(diff) < 0.000001:
            break
        beta_old = beta_new
    return beta_new

def predict(x, beta):
    p = cp(x, beta)
    if p > 0.5:
        return 1.
    else:
        return 0.

def zscore(X, Y, beta):
    w = W(X, beta)
    M = np.transpose(X).dot(w).dot(X)
    M = np.linalg.inv(M)
    
    zscores = np.zeros(len(beta))
    for i in range(len(zscores)):
        var = np.sqrt(M[i,i])
        zscores[i] = beta[i]/var
    
    return zscores

if __name__ == '__main__':
    inputs,Y = hd.loaddata()
    X = augment(inputs)[:,np.array([0,2,3,4,7])]
    print X.shape
    #print np.linalg.matrix_rank(X), X.shape
    beta = logistic(X, Y)
    error = 0
    for i in range(len(X)):
        res = predict(X[i], beta)
        if Y[i] != res:
            error += 1
    print 'error ratio:', float(error)/len(X)
    print beta
    print zscore(X,Y,beta)
