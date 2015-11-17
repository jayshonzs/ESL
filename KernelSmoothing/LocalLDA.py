'''
Created on 2014-6-22

@author: xiajie
'''
import numpy as np
from SupervisedBasic import zipcode

def weight(x0, x, lmbda):
    return np.exp(-np.transpose(x0-x).dot(x0-x)/(2*lmbda))

def cal_pi(x0, data_set, K, lmbda):
    pi = np.zeros(K)
    n = 0.
    for k in range(K):
        for i in range(len(data_set[k])):
            w = weight(x0, data_set[k][i], lmbda)
            pi[k] += w
            n += w
    return pi/N

def cal_mean(x0, data_set, K, lmbda):
    means = np.zeros((K,len(data_set[0][0])))
    for k in range(K):
        n = 0.
        for i in range(len(data_set[k])):
            x = data_set[k][i]
            w = weight(x0, x, lmbda)
            means[k] = means[k] + x*w
            n += w
        means[k] /= n
    return means

def cal_thegma(x0, data_set, means, K, N, lmbda):
    p = len(means[0])
    thegma = np.zeros((p,p))
    for k in range(K):
        g = np.zeros((p,p))
        for i in range(len(data_set[k])):
            x = data_set[k][i]
            w = weight(x0, x, lmbda)
            diff = x - means[k]
            g += w*np.outer(diff, diff)/(N-K)
        thegma += g
    return thegma

def cookdata(data):
    cls = {}
    inputs = data[:,1:].tolist()
    outputs = data[:,0].tolist()
    for i in range(len(inputs)):
        cls.setdefault(outputs[i],[])
        cls[outputs[i]].append(inputs[i])
    for k in cls.keys():
        cls[k] = np.array(cls[k])
    return cls

def discriminant(x, u, thegma, pi):
    inthegma = np.linalg.inv(thegma)
    return np.transpose(x).dot(inthegma).dot(u) - 0.5*np.transpose(u).dot(inthegma).dot(u) + np.log(pi)

def classify(x, U, T, P, K):
    dmax = -999999999
    index = 99
    for i in range(K):
        dis = discriminant(x, U[i], T, P[i])
        if dis > dmax:
            dmax = dis
            index = i
    return index

def run_classify(train, test, N):
    lmbda = 100
    error = 0.
    for i in range(len(test)):
        x0 = np.array(test[i][1:])
        y0 = test[i][0]
        pi = cal_pi(x0, train, len(train), lmbda)
        means = cal_mean(x0, train, len(train), lmbda)
        sigma = cal_thegma(x0, train, means, len(train), N, lmbda)
        
        res = classify(x0, means, sigma, pi, len(train))
        if res != y0:
            error += 1.
    print 'error rate:', error/len(test)

if __name__ == '__main__':
    train, test = zipcode.loaddata()
    N = len(train)
    train_cls = cookdata(train)
    #test_cls = cookdata(test)
    K = len(train_cls)
    run_classify(train_cls, test, N)
