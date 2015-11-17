'''
Created on 2014-7-21

@author: xiajie
'''
import numpy as np

'''
Useful for simple classifiers
'''

def simulate_data():
    kai = 9.34
    X = np.zeros((200,10))
    Y = np.zeros(200)
    for i in range(200):
        x = []
        for j in range(10):
            xj = np.random.standard_normal(1)[0]
            x.append(xj)
        x = np.array(x)
        X[i] = x
        if np.sum(x**2) > kai:
            Y[i] = 1.
        else:
            Y[i] = -1.
    return X, Y

def divideset(X, Y, W, col, val):
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    W1 = []
    W2 = []
    for i in range(len(X)):
        if X[i][col] <= val:
            X1.append(X[i])
            Y1.append(Y[i])
            W1.append(W[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])
            W2.append(W[i])
    return np.array(X1), np.array(X2), np.array(Y1), np.array(Y2), np.array(W1), np.array(W2)

def deviance(Y, W):
    n = len(Y)
    sw = np.sum(W)
    n0 = 0
    n1 = 0
    for i in range(n):
        if Y[i] == -1:
            n0 += W[i]
        else:
            n1 += W[i]
    p0 = float(n0)/sw
    p1 = float(n1)/sw
    d0 = 0
    if p0 > 0:
        d0 = p0*np.log(p0)
    d1 = 0
    if p1 > 0:
        d1 = p1*np.log(p1)
    d = 0 - d0 - d1
    return d

def missclassifyerror(Y, W):
    sw = np.sum(W)
    n0 = 0
    n1 = 0
    for i in range(len(Y)):
        if Y[i] == -1:
            n0 += W[i]
        else:
            n1 += W[i]
    if n0 >= n1:
        return float(n1)/sw
    else:
        return float(n0)/sw

def impurity(X, Y, col, val, W):
    X1, X2, Y1, Y2, W1, W2 = divideset(X, Y, W, col, val)
    if len(Y1) == 0 or len(Y2) == 0:
        return None
    d0 = missclassifyerror(Y1, W1)
    d1 = missclassifyerror(Y2, W2)
    #print 'col:', col, ' val:', val, ' d0:', d0, ' d1:', d1
    imp = (len(X1)*d0+len(X2)*d1, X1, X2, Y1, Y2, W1, W2)
    return imp

def weighted_stump(X, Y, W):
    min_deviance = 999999999.
    best_criteria = None
    best_sets = None
    column_count = len(X[0])
    for col in range(column_count):
        column = X[:,col]
        for s in range(len(column)):
            #print col, s
            res = impurity(X, Y, col, column[s], W)
            if res == None:
                continue
            q, X1, X2, Y1, Y2, W1, W2 = res
            total_impurity = len(X)*missclassifyerror(Y, W)
            #print 'running:', q, total_impurity
            if q < total_impurity and q < min_deviance:
                min_deviance = q
                best_criteria = (col, column[s])
                best_sets = [(X1, Y1, W1),(X2, Y2, W2)]
                #print 'min:', q
    return (best_criteria, best_sets)

def stump_predict(stump, x):
    cri = stump[0]
    sets = stump[1]
    if x[cri[0]] < cri[1]:
        Y = sets[0][1]
        W = sets[0][2]
    else:
        Y = sets[1][1]
        W = sets[1][2]
    negs = 0.
    for i in range(len(Y)):
        if Y[i] == -1.:
            negs += W[i]
    if negs > np.sum(W)/2.:
        return -1
    else:
        return 1

def error(stump, X, Y, W):
    sw = np.sum(W)
    N = len(Y)
    err = 0.
    for i in range(N):
        res = stump_predict(stump, X[i])
        if res != Y[i]:
            err += W[i]
    print 'error:', err, sw
    return err/sw

def reweight(W, X, Y, stump, alpha):
    new_W = np.zeros(len(W))
    for i in range(len(W)):
        res = stump_predict(stump, X[i])
        if res == Y[i]:
            new_W[i] = W[i]
        else:
            new_W[i] = W[i]*np.exp(alpha)
    return new_W

def boosting(X, Y, M=200):
    N = len(X)
    G = []
    W = np.ones(N)*(1./N)
    for m in range(M):
        print m
        stump = weighted_stump(X, Y, W)
        err_m = error(stump, X, Y, W)
        alpha_m = np.log((1-err_m)/err_m)
        print "error m:", err_m
        W = reweight(W, X, Y, stump, alpha_m)
        #print W
        G.append((stump, alpha_m))
    return G

def predict(G, x):
    last_res = 0.
    for g in G:
        res = stump_predict(g[0], x)
        last_res += res*g[1]
    return last_res

if __name__ == '__main__':
    X, Y = simulate_data()
    G = boosting(X, Y)
    err = 0.
    for i in range(len(Y)):
        res = predict(G, X[i])
        if res > 0:
            res = 1
        else:
            res = -1
        print res, Y[i]
        if res != Y[i]:
            err += 1.
    print 'error ratio:', err/len(Y)
