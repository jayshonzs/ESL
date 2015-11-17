'''
Created on 2014-6-18

@author: xiajie
'''
import numpy as np
from LinearClassification import QDA_4_vowel as qda
import PhonemeLogistic as pl

def gconverter(x):
    if x == 'aa':
        return 0
    elif x == 'ao':
        return 1
    elif x == 'sh':
        return 2
    elif x == 'iy':
        return 3
    elif x == 'dcl':
        return 4

def loaddata():
    inputs_cols = [i for i in range(1,257)]
    inputs = np.genfromtxt('phoneme.data', delimiter=',', skip_header=1, dtype=float, usecols=inputs_cols)
    outputs = np.genfromtxt('phoneme.data', delimiter=',', skip_header=1, dtype=float, converters={257:gconverter}, usecols=[257])
    return inputs, outputs

def createknots(start=1, end=256):
    l = np.arange(start,end,20).tolist()
    for i in range(len(l)):
        l[i] -= 3
    return l[1:]

def discriminant(x, u, sigma, pi):
    inthegma = np.linalg.inv(sigma)
    return -0.5*np.log(np.linalg.det(sigma)) - 0.5*np.transpose(x-u).dot(inthegma).dot(x-u) + np.log(pi)

def classify(x, U, S, P, K):
    dmax = -999999999
    index = 99
    for i in range(K):
        dis = discriminant(x, U[i], S[i], P[i])
        if dis > dmax:
            dmax = dis
            index = i
    return index+1.

def test(tests, n, U, S, P, K):
    err = 0.
    for key in tests.keys():
        for i in range(len(tests[key])):
            res = classify(tests[key][i], U, S, P, K)
            #print key, res
            if key != res:
                err += 1.
    return err/n

def split_data(data_in, data_out, i):
    ln = len(data_in)
    part_size = ln/10
    index = [j for j in range(ln)]
    test_in = data_in[i*part_size:(i+1)*part_size, :]
    test_out = data_out[i*part_size:(i+1)*part_size]
    train_index = index[0:i*part_size] + index[(i+1)*part_size:]
    train_in = data_in[train_index, :]
    train_out = data_out[train_index]
    return train_in, train_out, test_in, test_out

def cookdata(inputs, outputs):
    train_inputs = {}
    M = 7
    N = 0
    for i in range(len(outputs)):
        train_inputs.setdefault(outputs[i], [])
        train_inputs[outputs[i]].append(inputs[i].tolist())
    for key in train_inputs.keys():
        cls = np.array(train_inputs[key])
        knots = createknots(1,256)
        p = len(cls[0])
        H = pl.basismatrix(knots, p, M)
        XS = pl.transform(H, cls, M)
        train_inputs[key] = XS
        N += len(XS)
    return train_inputs, N, M

if __name__ == '__main__':
    data_in, data_out = loaddata()
    error = 0.
    for i in range(10):
        dtrain_in, dtrain_out, dtest_in, dtest_out = split_data(data_in, data_out, i)
        train, N, p = cookdata(dtrain_in, dtrain_out)
        K = len(train)
        pi = qda.cal_pi(train, K, N)
        means = qda.cal_means(train, K, p)
        sigmas = qda.cal_sigma(train, means, N, p)
        td, tn, tp = cookdata(dtest_in, dtest_out)
        e = test(td, tn, means, sigmas, pi, K)
        print e
        error += e
    print 'total:', error/10.
