'''
Created on 2014-6-15

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
from LinearClassification import Logistic as lg

def gconverter(x):
    if x == 'aa':
        return 0
    elif x == 'ao':
        return 1
    else:
        return 2

def ttconverter(x):
    segs = x.split('.')
    if segs[0] == 'train':
        return 0
    else:
        return 1
    
def cookdata(inputs, outputs, tt):
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []
    for i in range(len(outputs)):
        if outputs[i] == 0 or outputs[i] == 1:
            if tt[i] == 0:
                train_inputs.append(inputs[i])
                train_outputs.append(outputs[i])
            else:
                test_inputs.append(inputs[i])
                test_outputs.append(outputs[i])
    return np.array(train_inputs), np.array(train_outputs), np.array(test_inputs), np.array(test_outputs)

def loaddata():
    inputs_cols = [i for i in range(1,257)]
    inputs = np.genfromtxt('phoneme.data', delimiter=',', skip_header=1, dtype=float, usecols=inputs_cols)
    outputs = np.genfromtxt('phoneme.data', delimiter=',', skip_header=1, dtype=float, converters={257:gconverter}, usecols=[257])
    tt = np.genfromtxt('phoneme.data', delimiter=',', skip_header=1, dtype=float, converters={258:ttconverter}, usecols=[258])
    return cookdata(inputs, outputs, tt)

def createknots(start=1, end=256):
    l = np.arange(start,end,20).tolist()
    for i in range(len(l)):
        l[i] -= 3
    return l[1:]

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

def basismatrix(knots, p, M):
    H = np.zeros((p,M))
    for i in range(p):
        for j in range(M):
            H[i,j] = N(i+1, j, knots)
    return H

def transform(H, train_in, M):
    ln = len(train_in)
    HT = np.transpose(H)
    XS = np.zeros((ln, M))
    for i in range(ln):
        XS[i] = HT.dot(train_in[i])
    return XS

def plot(x):
    plt.plot([i for i in range(1,257)], x)
    plt.show()    

if __name__ == '__main__':
    M = 12
    train_in, train_out, test_in, test_out = loaddata()
    #plot(train_in[101])
    print train_in.shape
    #print outputs.shape
    knots = createknots(1,256)
    print knots
    p = len(train_in[0])
    H = basismatrix(knots, p, M)
    XS = transform(H, train_in, M)
    print XS.shape
    print np.linalg.matrix_rank(XS)
    beta = lg.logistic(XS, train_out)
    print beta
    error = 0
    TXS = transform(H, test_in, M)
    for i in range(len(TXS)):
        res = lg.predict(TXS[i], beta)
        if test_out[i] != res:
            error += 1
    print 'error ratio:', float(error)/len(TXS)
