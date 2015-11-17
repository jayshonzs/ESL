'''
Created on 2014-5-13

@author: xiajie
'''
import numpy as np
import prostate
import leastsquare

def backstepwise(data, out, k):
    feature_num = len(data[0])
    index_array = [i for i in range(feature_num)]
    droper = feature_num - k
    
    for i in range(droper):
        beta = leastsquare.ls(data[:,np.array(index_array)], out)
        yhat = leastsquare.predict(data[:,np.array(index_array)], beta)
        theg2 = leastsquare.thegama2(yhat, out, len(index_array))
        z = leastsquare.z_score(beta, np.sqrt(theg2), data[:,np.array(index_array)])
        min_z = 999999999.
        min_idx = None
        for j in range(len(z)):
            if z[j] < min_z:
                min_z = z[j]
                min_idx = j
        index_array.remove(index_array[min_idx])
    
    beta = leastsquare.ls(data[:,np.array(index_array)], out)
    
    return beta, index_array

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata(inputs, output, Ttype)
    beta, indexs = backstepwise(leastsquare.augment(train_data), train_out, 3)
    print beta
    print indexs
