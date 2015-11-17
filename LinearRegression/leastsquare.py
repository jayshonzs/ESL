'''
Created on 2014-5-4

@author: xiajie
'''
import numpy as np
import prostate

def ls(X, Y):
    #print X
    #print Y
    XT = np.transpose(X)
    M = XT.dot(X)
    if np.linalg.matrix_rank(M) < len(M):
        return None
    beta = np.linalg.inv(M).dot(XT).dot(Y)
    return beta

def augment(data):
    dlist = data.tolist()
    for i in range(len(dlist)):
        dlist[i].insert(0, 1)
    return np.array(dlist)

def predict(X, beta):
    yhat = X.dot(beta)
    return yhat

def RSS(X, Y, beta):
    yhat = predict(X, beta)
    return np.sum((yhat-Y)**2)

def thegama2(yhat, output, p):
    delta = yhat-output
    return np.sum(delta*delta)/(len(yhat)-p-1)

def z_score(beta, theg, data):
    z = np.zeros(len(beta))
    DT = np.transpose(data)
    M = np.linalg.inv(DT.dot(data))
    for i in range(len(beta)):
        z[i] = beta[i]/(theg*np.sqrt(M[i,i]))
    return z

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata2(inputs, output, Ttype)
    beta = ls(augment(train_data), train_out)
    print beta
    #yhat = predict(augment(train_data), beta)
    #theg2 = thegama2(yhat, train_out, len(train_data[0]))
    #print theg2
    #z = z_score(beta, np.sqrt(theg2), augment(train_data))
    #print z
    
    for i in range(len(test_out)):
        lst = test_data[i].tolist()
        lst.insert(0,1.)
        new_lst = np.array(lst)
        print predict(new_lst, beta), test_out[i]
    
    print RSS(augment(test_data), test_out, beta)/float(len(test_out))
    