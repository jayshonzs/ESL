'''
Created on 2014-8-12

@author: xiajie
'''
import numpy as np
import market_data as md
import itertools
import random

class features:
    def __init__(self, features=[], support=0., count=0.):
        self.features = features
        self.support = support
        self.count = count

def compare(f1, f2):
    if f1.features == f2.features:
        return True
    else:
        return False

def remix(lst1, lst2, k):
    combined = set(lst1+lst2)
    new_lsts = list(itertools.combinations(combined, k))
    for i in range(len(new_lsts)):
        new_lsts[i] = sorted(list(new_lsts[i]))
    return new_lsts

def cal_support(Z, new_fs):
    N = len(Z)
    supports = []
    for t in new_fs:
        count = 0.
        for i in range(N):
            flag = True
            for j in range(len(t)):
                if Z[i,t[j]] == 0:
                    flag = False
                    break
            if flag == True:
                count += 1.
        s = count/N
        supports.append(s)
    return supports

def apriori(Z, support=0.1, K=5):
    N = len(Z)
    P = len(Z[0])
    res = []
    res1 = []
    for p in range(P):
        new_f = features(features=[p])
        for i in range(N):
            if Z[i,p] == 1:
                new_f.count += 1.
        new_f.support = new_f.count/N
        if new_f.support >= support:
            res1.append(new_f)
    res.append(res1)
    
    print '1'
    
    for k in range(1,K):
        print '%d' % (k+1)
        res1 = []
        feats = res[k-1]
        #for feat in feats:
        #    print feat.features
        print 'feats len:', len(feats)
        for features1 in feats:
            for features2 in feats:
                if compare(features1, features2) == True:
                    continue
                f1 = features1.features
                f2 = features2.features
                new_fs = remix(f1, f2, k+1)
                #print new_fs
                new_supports = cal_support(Z, new_fs)
                for i, s in enumerate(new_supports):
                    if s > support:
                        r_flag = False
                        for item in res1:
                            if item.features == new_fs[i]:
                                r_flag = True
                                break
                        if r_flag == False:
                            res1.append(features(new_fs[i], s))
        res.append(res1)
    return res

if __name__ == '__main__':
    #print remix([3,4,5,6], [1,2,3,4], 5)
    Z = md.load()
    tmp = Z.tolist()
    random.shuffle(tmp)
    Z = np.array(tmp[:500])
    print Z.shape
    np.savetxt('Z.data', Z, fmt='%d', delimiter=' ')
    #np.savetxt(fname, X, fmt, delimiter, newline, header, footer, comments)
    supports = apriori(Z)
    last_sup = supports[-1]
    for i in range(len(last_sup)):
        print last_sup[i].features
        print last_sup[i].support
        print '**********'
    