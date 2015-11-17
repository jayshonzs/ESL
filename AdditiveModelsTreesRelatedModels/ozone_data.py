'''
Created on 2014-7-15

@author: xiajie
'''
import numpy as np
import random

def load():
    data = np.genfromtxt('ozone.data', dtype=float, skip_header=1)
    return data

def traintest(data):
    test_idx = []
    train_idx = []
    l = range(len(data))
    i = 0
    while True:
        if len(test_idx) == 30:
            break
        rd = random.choice(l)
        if rd not in test_idx:
            test_idx.append(rd)
        else:
            continue
    for i in range(len(data)):
        if i not in test_idx:
            train_idx.append(i)
    data_test = data[np.array(test_idx)]
    data_train = data[np.array(train_idx)]
    return data_train, data_test

def cook(data):
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

if __name__ == '__main__':
    data = load
    print data
    print data.shape
