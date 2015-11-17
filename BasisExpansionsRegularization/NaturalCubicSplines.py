'''
Created on 2014-6-11

@author: xiajie
'''
import numpy as np
from LinearClassification import Logistic as lg
from LinearClassification import SAHeart_data as hd

def d(x, k, knots):
    kdiff = knots[-1] - knots[k-1]
    item1 = x-knots[k-1]
    if item1 <= 0:
        item1 = 0
    item2 = x-knots[-1]
    if item2 <= 0:
        item2 = 0
    return (item1**3 - item2**3)/kdiff

def N(x, k, knots):
    k += 1
    if k == 1:
        return 1
    elif k == 2:
        return x
    else:
        return d(x, k-2, knots)-d(x, len(knots)-1, knots)

def split(data, K):
    Knots = np.zeros((len(data[0]), K))
    for i in range(len(data[0])):
        if i == 3:
            Knots[i] = np.zeros(K)
            continue
        feature = data[:,i]
        amin = feature.min()
        amax = feature.max()
        step = (amax - amin)/(K+1.)
        a = np.arange(amin, amax, step).tolist()
        a = a[1:]
        Knots[i] = np.array(a)
        #print amin, amax, a
    return Knots

def cookdata(data, K, knots):
    df = 1 + (K-1)*(len(data[0])-1) + 1
    L = len(data)
    new_matrix = np.zeros((L,df))
    for i in range(L):
        row = [1]
        for j in range(len(data[0])):
            seg = []
            if j == 3:
                seg.append(data[i,j])
                row = row + seg
                continue
            for k in range(0,K):
                seg.append(N(data[i,j], k, knots[j]))
            row = row + seg[1:]
        new_matrix[i] = np.array(row)
    return new_matrix

if __name__ == '__main__':
    K = 5
    X, Y = hd.loaddata()
    knots = split(X, K)
    print knots
    H = cookdata(X, K, knots)
    np.savetxt('H.dat', H, delimiter=',')
    print H.shape
    print np.linalg.matrix_rank(H)
    beta = lg.logistic(H, Y)
    print beta
    error = 0
    for i in range(len(H)):
        res = lg.predict(H[i], beta)
        if Y[i] != res:
            error += 1
    print 'error ratio:', float(error)/len(X)
