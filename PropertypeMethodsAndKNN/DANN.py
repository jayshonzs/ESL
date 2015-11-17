'''
Created on 2014-8-8

@author: xiajie
'''
import numpy as np

def euclid(x0, x1):
    return np.linalg.norm(x1-x0)

def adaptive_metric(x0, x1, sig):
    d = x1-x0
    return np.transpose(d).dot(sig).dot(d)

def nearest(X, classes, x0, k=50, sig=None):
    ret = []
    nns = None
    if sig == None:
        nns = [(euclid(x0, X[i]), i) for i in range(len(X))]
    else:
        nns = [(adaptive_metric(x0, X[i], sig), i) for i in range(len(X))]
    nns.sort()
    for nn in nns[:k]:
        ret.append((X[nn[1]], classes[nn[1]]))
    return ret

def Cov(X):
    p = len(X[0])
    C = np.zeros((p,p))
    means = np.mean(X, axis=0)
    for i in range(len(X)):
        diff = X[i]-means
        C += np.outer(diff, diff)
    return C

def getW(X, classes, K):
    p = len(X[0])
    W = np.zeros((p, p))
    
    PI = np.zeros(K)
    groups = {}
    for i in range(X):
        x = X[i]
        c = classes[i]
        PI[c] += 1.
        groups.setdefault(c, [])
        groups[c].append(x.tolist())
    PI /= float(len(X))
    for key in groups.keys():
        groups[key] = np.array(groups[key])
    
    for k in range(K):
        pi = PI[k]
        if pi > 0:
            W = W + pi*Cov(groups[k])
    
    return W

def getB(X, classes, K):
    p = len(X[0])
    B = np.zeros((p, p))
    x_mean = np.mean(X, axis=0)
    
    PI = np.zeros(K)
    groups = {}
    for i in range(len(X)):
        x = X[i]
        c = classes[i]
        PI[c] += 1.
        groups.setdefault(c, [])
        groups[c].append(x.tolist())
    PI /= float(len(X))
    for key in groups.keys():
        groups[key] = np.array(groups[key])
        groups[key] = np.mean(groups[key], axis=0)
    
    for k in range(K):
        pi = PI[k]
        if pi > 0:
            d = groups[k]-x_mean
            B = B + pi*np.outer(d, d)
    
    return B

def predict(X, classes, x0, eps=1.0):
    s = [classes]
    K = len(s)
    nns = nearest(X, classes, x0, 50)
    local_X = [nns[i][0].tolist() for i in range(len(nns))]
    local_X = np.array(local_X)
    local_C = [nns[i][1] for i in range(len(nns))]
    W = getW(local_X, local_C, K)
    B = getB(local_X, local_C, K)
    
    WI = np.linalg.inv(W)**0.5
    BS = WI.dot(B).dot(WI)
    sigma = WI.dot(BS+eps*np.identity(len(BS))).dot(WI)
    
    nns = nearest(local_X, local_C, x0, k=15, sig=sigma)
    
    vote = np.zeros(K)
    for nn in nns:
        vote[nn[1]] += 1
    max_v = 0
    best_i = None
    for i in range(K):
        if vote[i] > max_v:
            max_v = vote[i]
            best_i = i
    return best_i

if __name__ == '__main__':
    pass
