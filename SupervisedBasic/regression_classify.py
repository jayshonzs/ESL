'''
Created on 2014-4-29

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def loaddata():
    data1 = np.genfromtxt('sdata1.txt')
    data2 = np.genfromtxt('sdata2.txt')
    
    return data1, data2

def cookdata(data1, data2):
    X = np.concatenate((data1, data2), axis=0)
    ones = np.ones((len(X),1))
    X = np.concatenate((ones, X), axis=1)
    Y = np.zeros(len(data1)+len(data2))
    for i in range(len(data1)):
        Y[i] = 1
    return X, Y

def ls(X, Y):
    XT = np.transpose(X)
    M = XT.dot(X)
    beta = np.linalg.inv(M).dot(XT).dot(Y)
    return beta

def drawclass(X, Y, beta, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    two_min, two_max = X[:, 2].min()-0.1, X[:, 2].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        res = np.transpose(beta[1:]).dot(inputs[i])+beta[0]
        clss = 0
        if res >= 0.5:
            clss = 1
        z.append(clss)
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    
    plt.scatter(X[:,1], X[:,2], s=30, c=Y, cmap=mycm)
    
    plt.show()

if __name__ == '__main__':
    data1, data2 = loaddata()
    X, Y = cookdata(data1, data2)
    #print X
    #print Y
    beta = ls(X, Y)
    print beta
    
    drawclass(X, Y, beta, 200)