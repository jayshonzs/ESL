'''
Created on 2014-5-14

@author: xiajie
'''
import numpy as np
import prostate
import leastsquare
import correlation

class Item:
    def __init__(self, index, active=False):
        self.index = index
        self.active = active

def standardize(data):
    norms = np.zeros(len(data[0]))
    mean = np.mean(data, axis=0)
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j,i] = (data[j,i]-mean[i])
    for i in range(len(data[0])):
        norm = np.linalg.norm(data[:,i])
        norms[i] = norm
        for j in range(len(data)):
            data[j,i] = data[j,i]/norm
    return data,norms, mean

def direction(X, Y, beta, r):
    d = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(r)
    return d

def get_index(s):
    return s.index

def LAR(data, out, alpha=0.5):
    beta = []
    Y_means = np.ones(len(out))*np.mean(out)
    residual = out - Y_means
    active_index = []
    origin_index = [Item(i,False) for i in range(len(data[0]))]
    B = None
    
    while True:
        max_correlation = -999999999
        item = None;
        for it in origin_index:
            if it.active == True:
                continue
            co = correlation.correlation(data[:,it.index], residual)
            if co > max_correlation:
                max_correlation = co
                item = it
        if item == None:
            break
        item.active = True
        active_index.append(item)
        active_index = sorted(active_index, key=get_index)
        idx = 999999999
        for i in range(len(active_index)):
            if active_index[i].index == item.index:
                idx = i
                break
        beta.insert(idx, 0)
        indexes = [it.index for it in active_index]
        B = np.array(beta)
        d = direction(data[:,np.array(indexes)],out,B,residual)
        B = B + alpha*d
        for j in range(len(B)):
            if B[j] < 0.001 and B[j] > -0.001:
                it = active_index[j]
                origin_index[it.index].active = False
                del B[j]
                del active_index[j]
        residual = out - Y_means - data[:,np.array(indexes)].dot(B)
        if len(B) == 4:
            break
        beta = B.tolist()
    return B, Y_means, indexes

def predict(x, beta, Y_mean):
    return Y_mean + np.transpose(x).dot(beta)

if __name__ == '__main__':   
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = prostate.cookdata2(inputs, output, Ttype)
    X, norms, x_means = standardize(train_data)
    #print train_data[:,1]
    #print sum(train_data[:,1])
    #print np.linalg.norm(train_data[:,1])
    B, Y_means, indexs = LAR(X, train_out, 1.)
    print B
    print indexs
    RSS = 0
    for i in range(len(test_out)):
        x = test_data[i]
        test_x = (x[np.array(indexs)]-x_means[np.array(indexs)])/norms[np.array(indexs)]
        print test_out[i], predict(test_x, B, Y_means[0])
        RSS += (test_out[i] - predict(test_x, B, Y_means[0]))**2
    print 'RSS:',RSS/len(test_data)