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
    centers1 = np.genfromtxt('scenters1.txt')
    centers2 = np.genfromtxt('scenters2.txt')
    
    return data1, data2, centers1, centers2

def cookdata(data1, data2, centers1, centers2):
    X = np.concatenate((data1, data2), axis=0)
    Y = np.zeros(len(data1)+len(data2))
    for i in range(len(data1)):
        Y[i] = 1
    C = np.concatenate((centers1, centers2), axis=0)
    return X, Y, C

def distance(x1, x2):
    return np.linalg.norm((x1-x2))

def bayes_classify(x0, C):
    mind = 999999999.
    idx = 0
    for i in range(len(C)):
        d = distance(x0, C[i])
        if d < mind:
            mind = d
            idx = i
    if idx < 10:
        return 1
    else:
        return 0

def drawclass(X, Y, C, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    two_min, two_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        res = bayes_classify(inputs[i], C)
        z.append(res)
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    
    plt.scatter(X[:,0], X[:,1], s=30, c=Y, cmap=mycm)
    
    plt.show()

if __name__ == '__main__':
    data1, data2, centers1, centers2 = loaddata()
    X, Y, C = cookdata(data1, data2, centers1, centers2)
    print X
    print Y
    print C
    drawclass(X, Y, C, 50)