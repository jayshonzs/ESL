'''
Created on 2014-5-4

@author: xiajie
'''
import numpy as np
import regression_classify
import knn

def loaddata():
    train_data = np.genfromtxt('zip.train')
    test_data = np.genfromtxt('zip.test')
    
    return train_data, test_data

def cookdata(data):
    dlist = data.tolist()
    cooked = []
    for i in range(len(dlist)):
        if data[i,0] == 2. or data[i,0] == 3.:
            cooked.append(dlist[i])
    t = [l[0] for l in cooked]
    zips = [l[1:] for l in cooked]
    for i in range(len(zips)):
        zips[i].insert(0,1)
    return np.array(t),np.array(zips)

def runtest(beta, test_data):
    t, zips = cookdata(test_data)
    correct = 0
    for i in range(len(zips)):
        res = np.transpose(beta).dot(zips[i])
        print t[i], res
        if t[i] == 2.:
            if res <= 2.5:
                correct += 1
        elif t[i] == 3.:
            if res > 2.5:
                correct += 1
    return 1.-(float(correct)/len(zips))

'''
k nearest neighbor
'''
def cookdata_knn(data):
    dlist = data.tolist()
    cooked = []
    for i in range(len(dlist)):
        if data[i,0] == 2. or data[i,0] == 3.:
            cooked.append(dlist[i])
    t = [l[0] for l in cooked]
    zips = [l[1:] for l in cooked]
    return np.array(t),np.array(zips)

def runtest_knn(train_data, test_data, k=1):
    train_t, train_zips = cookdata_knn(train_data)
    test_t, test_zips = cookdata_knn(test_data)
    correct = 0
    for i in range(len(test_zips)):
        res = knn.knn(train_zips, train_t, k, test_zips[i])
        if res == test_t[i]:
            correct += 1
    return 1.-(float(correct)/len(test_zips))

if __name__ == '__main__':
    train_data, test_data = loaddata()
    print train_data.shape
    t, zips = cookdata(train_data)
    beta = regression_classify.ls(zips, t)
    print 'ls test error: %f' % runtest(beta, test_data)
    
    print 'knn1 test error: %f' % runtest_knn(train_data, test_data, 1)
    print 'knn5 test error: %f' % runtest_knn(train_data, test_data, 5)
    print 'knn10 test error: %f' % runtest_knn(train_data, test_data, 10)
    print 'knn15 test error: %f' % runtest_knn(train_data, test_data, 15)
