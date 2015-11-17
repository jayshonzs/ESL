'''
Created on 2014-5-17

@author: xiajie
'''
import numpy as np
import prostate

def center_data(X, Y, normalize=False):
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y)
    X_std = None
    X_ret = np.zeros(X.shape)
    #Y_ret = np.zeros(Y.shape)
    for i in range(len(X[0])):
        for j in range(len(X)):
            X_ret[j,i] = X[j,i] - X_mean[i]
    if normalize == True:
        X_std = np.std(X, axis=0)
        for i in range(len(X[0])):
            X_ret[:,i] = X_ret[:,i]/X_std[i]
    #Y_ret = Y - np.ones(len(Y))*Y_mean
    return X_ret, X_mean, Y_mean, X_std

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    #train_data, train_out, test_data, test_out = prostate.cookdata(inputs, output, Ttype)
    X, X_mean, Y_mean, std = center_data(inputs, output)
    print np.sum(X, axis=0)
    print X_mean
    print Y_mean