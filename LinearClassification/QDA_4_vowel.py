'''
Created on 2014-6-10

@author: xiajie
'''
import numpy as np
import vowel_data as vw

def cookdata(inputs, outputs):
    classes = {}
    for i in range(len(outputs)):
        classes.setdefault(outputs[i],[])
        classes[outputs[i]].append(inputs[i].tolist())
    for key in classes.keys():
        classes[key] = np.array(classes[key])
    return classes

def cal_pi(data_set, K, N):
    pi = np.zeros(K)
    for k in data_set.keys():
        pi[k-1] = len(data_set[k])/float(N)
    return pi

def cal_means(data_set, K, p):
    means = np.zeros((K,p))
    for k in data_set.keys():
        means[k-1] = np.mean(data_set[k], axis=0)
    return means

def cal_sigma(data_set, means, N, p):
    thegmas = []
    for k in data_set.keys():
        g = np.zeros((p,p))
        n = len(data_set[k])
        for i in range(n):
            x = data_set[k][i]
            diff = x - means[k-1]
            g += np.outer(diff, diff)/(n-1)
        thegmas.append(g)
    return thegmas

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

def test(test_input, test_output, U, S, P, K):
    err = 0.
    for i in range(len(test_input)):
        res = classify(test_input[i], U, S, P, K)
        print test_output[i], res
        if test_output[i] != res:
            err += 1.
    print 'error rate:', err/len(test_output)

if __name__ == '__main__':
    train_input,train_output,test_input,test_output = vw.loaddata()
    N = len(train_input)
    p = len(train_input[0])
    classes = cookdata(train_input, train_output)
    K = len(classes)
    pi = cal_pi(classes, K, N)
    means = cal_means(classes, K, p)
    sigmas = cal_sigma(classes, means, N, p)
    test(test_input, test_output, means, sigmas, pi, K)
