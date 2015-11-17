'''
Created on 2014-7-12

@author: xiajie
'''
import numpy as np

def cook(X, Y, train_test):
    X = X.tolist()
    Y = Y.tolist()
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(len(X)):
        if train_test[i] == 0:
            X_train.append(X[i])
            Y_train.append(Y[i][0])
        else:
            X_test.append(X[i])
            Y_test.append(Y[i][0])
    return X_train, X_test, Y_train, Y_test

def load():
    data = np.genfromtxt('spam.data', dtype=float)
    train_test = np.genfromtxt('spam.traintest', dtype=int)
    return cook(data[:,:57], data[:,57:], train_test)

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load()
    print Y_test
