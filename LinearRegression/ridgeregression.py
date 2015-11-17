'''
Created on 2014-5-13

@author: xiajie
'''
import numpy as np
import prostate
#import leastsquare
import preprocess

def ridgeregression(X, Y, lanbda):
    M = np.transpose(X).dot(X)
    N = np.identity(len(M))*lanbda
    MN = M + N
    inv = np.linalg.inv(MN)
    inter = inv.dot(np.transpose(X))
    #print '@@@@@@@@@@@@@@@'
    #for i in range(8):
    #    for j in range(67):
    #        print inter[i,j],
    #    print '.'
    #print '@@@@@@@@@@@@@@@'
    beta = inter.dot(Y)
    return beta

def df(X, lanbda):
    M = np.transpose(X).dot(X)
    H = X.dot(np.linalg.inv(M+np.identity(len(M))*lanbda)).dot(np.transpose(X))
    return np.trace(H)

def predict(x, beta):
    #print X_mean[2],x[2]
    X = x
    #print X
    return np.transpose(X).dot(beta)

if __name__ == '__main__':
    lanbda = 23
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata2(inputs, output, Ttype)
    X, X_mean, Y_mean, X_std = preprocess.center_data(train_data, train_out, True)
    print df(X, lanbda)
    beta = ridgeregression(X, train_out, lanbda)
    #for i in range(len(X)):
    #    print X[i]
    #print '***************************'
    #for i in range(len(train_out)):
    #    print train_out[i]
    #print '***************************'
    #beta2 = ridgeregression(X, Y, lanbda)
    print beta
    #beta0 = Y_mean - X_mean.dot(beta)
    #new_beta = beta.tolist()
    #new_beta.insert(0,beta0)
    #print new_beta
    #print beta2 #same
    if 0:
        RSS = 0
        for i in range(len(test_data)):
            #lst = test_data[i].tolist()
            #lst.insert(0,1.)
            #new_lst = np.array(lst)
            #print "1:",test_data[i]
            standardized = (test_data[i]-X_mean)/X_std
            #print "2:",standardized
            print test_out[i], predict(standardized, beta)+Y_mean
            RSS += (test_out[i]-predict(standardized, beta)-Y_mean)**2
        RSS /= len(test_data)
        print RSS
    
    #print leastsquare.RSS(leastsquare.augment(test_data), test_out, beta)/float(len(test_out))