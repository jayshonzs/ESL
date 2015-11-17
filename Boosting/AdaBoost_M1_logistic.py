'''
Created on 2014-7-20

@author: xiajie
'''
import numpy as np
from LinearClassification import SAHeart_data as hd

'''
It seems not good for logistic.
'''

def cp(x, beta):
    lf = x.dot(beta)
    #print '************'
    #print x
    #print lf
    #print beta
    return np.exp(lf)/(1+np.exp(lf))

def D(X, beta, W):
    d = np.zeros(len(X))
    for i in range(len(X)):
        pi = cp(X[i],beta)
        #if type(pi) is types.ListType:
        #    print X
        d[i] = -1.*W[i]*pi*(1-pi)
    return np.diag(d)

def Hessian(X, beta, W, lmbda):
    d = D(X, beta, W)
    return np.transpose(X).dot(d).dot(X)-lmbda*np.identity(len(X[0]))

def Z(X, Y, W, beta):
    N = len(X)
    z = np.zeros(N)
    for i in range(N):
        z[i] = W[i]*(Y[i]-cp(X[i], beta))
    return z

def differential(z, X, beta, lmbda):
    lmbdas = np.ones(len(X[0]))
    lmbdas *= lmbda
    return np.transpose(X).dot(z)-lmbdas

def weighted_logistic(X, W, lmbda=0.0001):    
    beta_old = np.zeros(len(X[0]))
    while True:
        h = Hessian(X, beta_old, W, lmbda)
        z = Z(X, Y, W, beta_old)
        deriv = differential(z, X, beta_old, lmbda)
        beta_new = beta_old - np.linalg.inv(h).dot(deriv)
        diff = np.sum((beta_new-beta_old)**2)
        if np.sqrt(diff) < 0.000001:
            break
        beta_old = beta_new
    return beta_new

def log_predict(x, beta):
    p = cp(x, beta)
    if p > 0.5:
        return 1.
    else:
        return 0.

def error(W, X, Y, beta):
    sw = np.sum(W)
    err = 0.
    for i in range(len(X)):
        res = log_predict(X[i], beta)
        if Y[i] != res:
            err += W[i]
    return err/sw

def reweight(W, X, Y, beta, alpha):
    new_W = np.zeros(len(W))
    for i in range(len(W)):
        res = log_predict(X[i], beta)
        if res == Y[i]:
            new_W[i] = W[i]
        else:
            new_W[i] = W[i]*np.exp(alpha)
    return new_W

def boosting(X, Y, M=100):
    N = len(X)
    G = []
    W = np.array([1./N for i in range(N)])
    for m in range(M):
        beta = weighted_logistic(X, W)
        err_m = error(W, X, Y, beta)
        alpha_m = np.log((1-err_m)/err_m)
        #print "error m:", err_m, "alpha m:", alpha_m
        W = reweight(W, X, Y, beta, alpha_m)
        #print W
        G.append((beta, alpha_m))
    return G

def predict(G, x):
    last_res = 0.
    for item in G:
        res = log_predict(x, item[0])
        last_res += res*item[1]
    return last_res

def augment(data):
    dlist = data.tolist()
    for i in range(len(dlist)):
        dlist[i].insert(0, 1.)
    return np.array(dlist)

def modify(Y):
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1

if __name__ == '__main__':
    inputs,Y = hd.loaddata()
    X = augment(inputs)[:,np.array([0,2,3,4,7])]
    G = boosting(X, Y)
    err = 0.
    for i in range(len(X)):
        res = predict(G, X[i])
        print res, Y[i]
        #if res != Y[i]:
        #    err += 1.
    #print "error ratio:", err/len(X)
