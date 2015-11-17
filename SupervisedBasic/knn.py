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
    Y = np.zeros(len(data1)+len(data2))
    for i in range(len(data1)):
        Y[i] = 1
    return X, Y

def distance(x1, x2):
    return np.linalg.norm((x1-x2))

def knn(X, Y, k, x0):
    min_y = Y.min()
    max_y = Y.max()
    dist = [(distance(x0, X[i]), i) for i in range(len(X))]
    dist.sort()
    #print dist
    sumofv = 0
    for j in range(k):
        item = dist[j]
        sumofv += Y[item[1]]
    if float(sumofv)/k >= (min_y+max_y)/2.0:
        return max_y
    else:
        return min_y

def drawclass(X, Y, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    two_min, two_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        res = knn(X, Y, 15, inputs[i])
        z.append(res)
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    
    plt.scatter(X[:,0], X[:,1], s=30, c=Y, cmap=mycm)
    
    plt.show()

if __name__ == '__main__':
    data1, data2 = loaddata()
    X, Y = cookdata(data1, data2)
    drawclass(X, Y, 50)