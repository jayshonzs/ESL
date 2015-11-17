'''
Created on 2014-5-4

@author: xiajie
'''
import numpy as np
import prostate

def doqr(data, output):
    q, r = np.linalg.qr(data)
    beta = np.linalg.inv(r).dot(np.transpose(q)).dot(output)
    return beta
    
def augment(data):
    dlist = data.tolist()
    for i in range(len(dlist)):
        dlist[i].insert(0, 1)
    return np.array(dlist)

if __name__ == '__main__':
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata(inputs, output, Ttype)
    beta = doqr(augment(train_data),train_out)
    print beta
