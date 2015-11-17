'''
Created on 2014-6-17

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import leastsquare as ls

def simulatedata(n):
    X = np.sort(np.random.random(n))
    Y = np.zeros(n)
    
    for i in range(n):
        e = np.random.normal(0., 0.35, 1)
        Y[i] = np.sin(X[i]*2.) + e
    
    return X, Y

def draw0(X, Y):
    plt.plot(X, Y)
    plt.show()
    
def augment(X):
    lst = X.tolist()
    ret = np.zeros((len(lst),2))
    for i in range(len(lst)):
        ret[i,0] = 1
        ret[i,1] = lst[i]
    return ret
    
def draw_linear_pwv(train_in, train_out):
    p = 1
    X = augment(train_in)
    XT = np.transpose(X)
    
    beta = ls.ls(X, train_out)
    
    Yhat = ls.predict(X, beta)
    thegsq = ls.thegama2(Yhat, train_out, p)
    beta_var = np.linalg.inv(XT.dot(X))*thegsq
    
    print beta_var
    
    PWV = []
    for i in range(len(X)):
        PWV.append(np.transpose(X[i]).dot(beta_var).dot(X[i]))
        
    #print PWV
    
    plt.plot(train_in, PWV)

def polynomial(X):
    lst = X.tolist()
    ret = np.zeros((len(lst),4))
    for i in range(len(lst)):
        ret[i,0] = 1
        ret[i,1] = lst[i]
        ret[i,2] = lst[i]**2
        ret[i,3] = lst[i]**3
    return ret
    
def draw_polynomial_pwv(train_in, train_out):
    p = 3
    X = polynomial(train_in)
    XT = np.transpose(X)
    
    beta = ls.ls(X, train_out)
    
    Yhat = ls.predict(X, beta)
    thegsq = ls.thegama2(Yhat, train_out, p)
    beta_var = np.linalg.inv(XT.dot(X))*thegsq
    
    print beta_var
    
    PWV = []
    for i in range(len(X)):
        PWV.append(np.transpose(X[i]).dot(beta_var).dot(X[i]))
        
    #print PWV
    
    plt.plot(train_in, PWV)
    
def cubic_h(x, k, knots):
    if k == 1:
        return 1
    elif k == 2:
        return x
    elif k == 3:
        return x**2
    elif k == 4:
        return x**3
    else:
        diff = x - knots[k-5]
        if diff > 0:
            return diff**3
        else:
            return 0

def cubic(X):
    knots = [0.33, 0.66]
    lst = X.tolist()
    ret = np.zeros((len(lst),6))
    for i in range(len(lst)):
        for j in range(6):
            ret[i, j] = cubic_h(lst[i], j+1, knots)
    return ret

def draw_cubic_spline_pwv(train_in, train_out):
    p = 5
    X = cubic(train_in)
    #print X.shape
    XT = np.transpose(X)
    #print X
    
    beta = ls.ls(X, train_out)
    
    Yhat = ls.predict(X, beta)
    thegsq = ls.thegama2(Yhat, train_out, p)
    beta_var = np.linalg.inv(XT.dot(X))*thegsq
    
    print beta_var
    
    PWV = []
    for i in range(len(X)):
        PWV.append(np.transpose(X[i]).dot(beta_var).dot(X[i]))
        
    #print PWV
    
    plt.plot(train_in, PWV)

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
    if k == 1:
        return 1
    elif k == 2:
        return x
    else:
        return d(x, k-2, knots)-d(x, len(knots)-1, knots)

def nscookdata(data, K, knots):
    df = 6
    L = len(data)
    new_matrix = np.zeros((L,df))
    for i in range(L):
        row = []
        for k in range(1,K+1):
            row.append(N(data[i], k, knots))
        new_matrix[i] = np.array(row)
    return new_matrix
    
def draw_natural_spline_pwv(train_in, train_out):
    p = 5
    knots = np.arange(0.1, 0.9, 0.16).tolist()
    knots.append(0.9)
    X = nscookdata(train_in, 6, knots)
    #print H
    #print H.shape
    XT = np.transpose(X)
    
    beta = ls.ls(X, train_out)
    
    Yhat = ls.predict(X, beta)
    thegsq = ls.thegama2(Yhat, train_out, p)
    beta_var = np.linalg.inv(XT.dot(X))*thegsq
    
    print beta_var
    
    PWV = []
    for i in range(len(X)):
        PWV.append(np.transpose(X[i]).dot(beta_var).dot(X[i]))
        
    #print PWV
    
    plt.plot(train_in, PWV)

if __name__ == '__main__':
    X, Y = simulatedata(50)
    #draw0(X, Y)
    draw_linear_pwv(X, Y)
    draw_polynomial_pwv(X, Y)
    draw_cubic_spline_pwv(X, Y)
    draw_natural_spline_pwv(X, Y)
    plt.show()
