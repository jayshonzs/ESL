'''
Created on 2014-5-7

@author: xiajie
'''
import numpy as np
import prostate
import leastsquare

def LB(origin, order):
    lst = []
    length = len(origin)
    
    for i in range(1,order):
        items = origin[:length-i] + origin[length-i+1:]
        print items
        lst.append(items)
        lst = lst + LB(items, i)
    
    return lst

def bestsubset(data, out, k):
    #print data.shape
    m = len(data[0])
    origin = [i+1 for i in range(m)]
    lst = []
    lst.append(origin)
    for i in range(m):
        items = origin[:m-i-1] + origin[m-i:]
        lst.append(items)
        print items
        lst = lst + LB(items, i+1)
    print lst
    min_rss = 999999999
    min_beta = None
    min_idx = None
    for item in lst:
        if len(item) == k:
            ones = np.ones(k,dtype=int)
            d = data[:, np.array(item)-ones]
            beta = leastsquare.ls(d, out)
            rss = RSS(d, out, beta)
            if rss < min_rss:
                min_rss = rss
                min_beta = beta
                min_idx = item
    return min_beta, min_idx

def RSS(X, Y, beta):
    yhat = leastsquare.predict(X, beta)
    return np.sum((yhat-Y)**2)

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata(inputs, output, Ttype)
    beta, idx = bestsubset(leastsquare.augment(train_data), train_out, 3)
    print beta
    print idx
