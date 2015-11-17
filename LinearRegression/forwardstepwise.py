'''
Created on 2014-5-13

@author: xiajie
'''
import numpy as np
import prostate
import leastsquare

def stepwise(data, out, k):
    feature_num = len(data[0])
    index_array = [0]
    indexs = [i for i in range(1,feature_num)]
    
    #beta = leastsquare.ls(data[:,np.array(index_array)], out)
    
    min_beta = None
    min_idx =None
    for i in range(1,k):
        min_rss = 999999999
        for idx in indexs:
            ia = index_array + [idx]
            beta = leastsquare.ls(data[:,ia], out)
            rss = RSS(data[:,ia], out, beta)
            if rss < min_rss:
                min_rss = rss
                min_beta = beta
                min_idx = idx
        index_array.append(min_idx)
        indexs.remove(min_idx)
    return min_beta, index_array

def RSS(X, Y, beta):
    yhat = leastsquare.predict(X, beta)
    return np.sum((yhat-Y)**2)

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata(inputs, output, Ttype)
    beta, indexs = stepwise(leastsquare.augment(train_data), train_out, 3)
    print beta
    print indexs