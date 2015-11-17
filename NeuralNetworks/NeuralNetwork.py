'''
Created on 2014-7-25

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
'''
One layer.
'''
def loaddata():
    data1 = np.genfromtxt('sdata1.txt')
    data2 = np.genfromtxt('sdata2.txt')
    
    return data1, data2

def standardize(data):
    sdata = np.zeros(data.shape)
    for i in range(len(data[0])):
        column = data[:,i]
        EX = np.sum(column)/len(column)
        EX2 = np.sum(column*column)/len(column)
        V = np.sqrt(EX2-EX*EX)
        for j in range(len(column)):
            sdata[j, i] = (data[j, i]-EX)/V
    return sdata

def cookdata(data1, data2):
    X = np.concatenate((data1, data2), axis=0)
    Y = np.zeros(len(data1)+len(data2))
    for i in range(len(data1)):
        Y[i] = 1
    return X, Y

def augment(data):
    dlist = data.tolist()
    for i in range(len(dlist)):
        dlist[i].insert(0, 1)
    return np.array(dlist)

def sigmoid(v):
    return 1./(1+np.exp(-v))

def dsigmoid(v):
    return -1.*np.exp(-v)/(1+np.exp(-v))**2

def CD(X, Y, f):
    N = len(X)
    D = np.zeros((2,N))
    for k in range(2):
        for i in range(N):
            if Y[i] == k:
                y = 1
            else:
                y = 0
            res = -2.*(y-f[i][k])
            D[k,i] = res
    return D

def CS(X, D, alpha, beta):
    N = len(X)
    S = np.zeros((10,N))
    for m in range(10):
        for i in range(N):
            sig = dsigmoid(np.transpose(alpha[m]).dot(X[i]))
            s = 0.
            for k in range(2):
                s += beta[k][m]*D[k,i]
            s = sig*s
            S[m, i] = s
    return S

def back_propagation(X, Y):
    P = len(X[0])
    N = len(X)
    #initialization
    alpha = np.zeros((10,P))
    beta = np.zeros((2,11))
    for i in range(len(alpha)):
        for j in range(len(alpha[0])):
            alpha[i,j] = random.random()/10.
    for k in range(len(beta)):
        for m in range(len(beta[0])):
            beta[k,m] = random.random()/10.
    #run
    while True:
        #forward
        f = cal(X, alpha, beta)
        #backward
        new_alpha = np.zeros((10,P))
        new_beta = np.zeros((2,11))
        D = CD(X, Y, f)
        S = CS(X, D, alpha, beta)
        for k in range(2):
            for m in range(10):
                dsum = 0.
                for i in range(N):
                    dsum += D[k,i]*CZ(X[i], alpha[m])
                new_beta[k,m] = beta[k,m] - 0.02*dsum
        for m in range(10):
            for l in range(P):
                dsum = 0.
                for i in range(N):
                    dsum += S[m,i]*X[i,l]
                new_alpha[m,l] = alpha[m,l] - 0.02*dsum
        
        alpha_diff = new_alpha - alpha
        beta_diff = new_beta - beta
        if np.linalg.norm(alpha_diff) < 0.0001 and np.linalg.norm(beta_diff) < 0.0001:
            break
        alpha = new_alpha
        beta = new_beta
    return alpha, beta

def cal(X, alpha, beta):
    N = len(X)
    res = []
    for i in range(N):
        T = classify(X[i], alpha, beta)
        res.append(T)
    return res

def classify(x, alpha, beta):
    Z = np.zeros(11)
    Z[0] = 1
    T = np.zeros(2)
    for m in range(10):
        Z[m+1] = sigmoid(np.transpose(alpha[m]).dot(x))
    for k in range(2):
        T[k] = np.transpose(Z).dot(beta[k])
    return T

def CZ(x, alpha):
    z = sigmoid(np.transpose(alpha).dot(x))
    return z

def predict(x, alpha, beta):
    res = classify(x, alpha, beta)
    if res[0] > res[1]:
        return 0
    else:
        return 1

def drawclass(X, Y, alpha, beta, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    two_min, two_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        res = predict(inputs[i], alpha, beta)
        z.append(res)
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    
    plt.scatter(X[:,0], X[:,1], s=30, c=Y, cmap=mycm)
    
    plt.show()

if __name__ == '__main__':
    data1, data2 = loaddata()
    X, Y = cookdata(data1, data2)
    alpha, beta = back_propagation(augment(X), Y)
    print alpha
    print '****************************'
    print beta
