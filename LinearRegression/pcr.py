'''
Created on 2014-5-25

@author: xiajie
'''
import numpy as np
import prostate
import preprocess

def theta(z, y):
    return np.transpose(z).dot(y)/np.transpose(z).dot(z)

def PCR(X,Y,m=7):
    z = []
    thetas = []
    beta = np.zeros(len(X[0]))
    U, S, V = np.linalg.svd(X, full_matrices=True)
    for i in range(m):
        z.append(X.dot(V[i]))
    for i in range(m):
        thetas.append(theta(z[i],Y))
    for i in range(m):
        beta = beta + V[i]*thetas[i]
    return beta

def predict(x, beta, Y_mean):
    return Y_mean + np.transpose(x).dot(beta)

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata2(inputs, output, Ttype)
    X, X_mean, Y_mean, X_std = preprocess.center_data(train_data, train_out, True)
    beta = PCR(X,train_out)
    print beta
    RSS = 0
    for i in range(len(test_out)):
        x = test_data[i]
        x = (x-X_mean)/X_std
        print test_out[i], predict(x,beta,Y_mean)
        RSS += (test_out[i]-predict(x,beta,Y_mean))**2
    print RSS/len(test_out)