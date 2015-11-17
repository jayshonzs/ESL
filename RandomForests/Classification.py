'''
Created on 2014-8-19

@author: xiajie
'''
import numpy as np
import random

class decisionnode:
    def __init__(self, col=-1, value=None, result=None, left_child=None, right_child=None, sets=None):
        self.col = col
        self.value = value
        self.result = result
        self.left_child = left_child
        self.right_child = right_child
        self.sets = sets
    
    def equal(self, other):
        if self.col != other.col:
            return False
        if self.value != other.value:
            return False
        if self.result != other.result:
            return False
        if len(self.sets[0]) != len(other.sets[0]):
            return False
        return True
    
def bootstrap(X, Y):
    N = len(X)
    P = len(X[0])
    Zx = np.zeros((N,P))
    Zy = np.zeros(N)
    for i in range(N):
        idx = random.randint(0,N-1)
        Zx[i] = X[idx]
        Zy[i] = Y[idx]
    return (Zx, Zy)

def divideset(X, Y, col, val):
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    for i in range(len(X)):
        if X[i][col] <= val:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])
    return X1, X2, Y1, Y2

def deviance(X, Y):
    n = len(X)
    n0 = 0
    n1 = 0
    for i in range(n):
        if Y[i] == 0:
            n0 += 1
        else:
            n1 += 1
    p0 = float(n0)/n
    p1 = float(n1)/n
    d0 = 0
    if p0 > 0:
        d0 = p0*np.log(p0)
    d1 = 0
    if p1 > 0:
        d1 = p1*np.log(p1)
    d = 0 - d0 - d1
    return d

def missclassifyerror(tree_node):
    y = tree_node.sets[1]
    n0 = 0
    n1 = 0
    for i in range(len(y)):
        if y[i] == 0:
            n0 += 1
        else:
            n1 += 1
    if tree_node.result == 0:
        return float(n1)/len(y)
    else:
        return float(n0)/len(y)
    
def impurity(X, Y, col, val):
    X1, X2, Y1, Y2 = divideset(X, Y, col, val)
    if len(Y1) == 0 or len(Y2) == 0:
        return None
    d0 = deviance(X1, Y1)
    d1 = deviance(X2, Y2)
    imp = (len(X1)*d0+len(X2)*d1, X1, X2, Y1, Y2)
    return imp

def clss(Y):
    n0 = 0
    for item in Y:
        if item == 0:
            n0 += 1
    #print n0, len(Y)
    if n0/len(Y) >= 0.5:
        return 0
    else:
        return 1
    
def buildrandomtree(X, Y, m):
    if len(X) == 0:
        return None
    elif len(X) <= 5:
        return decisionnode(result=clss(Y), sets=(X,Y))
    
    min_deviance = 999999999999.0
    best_criteria = None
    best_sets = None
    
    rd_cols = [i for i in range(len(X[0]))]
    rd_cols = random.shuffle(rd_cols)[:m]
    rd_cols.sort()
    
    for col in rd_cols:
        column = [X[i][col] for i in range(len(X))]
        #print column
        for s in range(len(column)):
            res = impurity(X, Y, col, column[s])
            if res == None:
                continue
            q, X1, X2, Y1, Y2 = res
            total_impurity = len(X)*deviance(X, Y)
            if q < total_impurity and q < min_deviance:
                min_deviance = q
                best_criteria = (col, column[s])
                best_sets = [(X1, Y1),(X2, Y2)]
    if best_sets == None:
        return decisionnode(result=clss(Y), sets=(X,Y))
    elif len(best_sets[0][0]) > 0 and len(best_sets[1][0]) > 0:
        leftBranch = buildrandomtree(best_sets[0][0], best_sets[0][1], m)
        rightBranch = buildrandomtree(best_sets[1][0], best_sets[1][1], m)
        return decisionnode(col=best_criteria[0], value=best_criteria[1], left_child=leftBranch, right_child=rightBranch, sets=(X,Y))
    else:
        return decisionnode(result=clss(Y), sets=(X,Y))

def buildforest(X, Y, B, m):
    forest = []
    for b in range(B):
        Zx, Zy = bootstrap(X, Y)
        rd_tree = buildrandomtree(Zx, Zy, m)
        forest.append(rd_tree)
    return forest

def tree_predict(obs, tree):
    if tree == None:
        return None
    tree_node = tree
    while True:
        if tree_node.left_child == None and tree_node.right_child == None:
            return tree_node.result
        col = tree_node.col
        val = tree_node.value
        if obs[col] <= val:
            tree_node = tree_node.left_child
        else:
            tree_node = tree_node.right_child

def predict(obs, forest):
    c0 = 0.
    c1 = 0.
    for tree in forest:
        res = tree_predict(obs, tree)
        if res == 0:
            c0 += 1.
        else:
            c1 += 1.
    if c0 >= c1:
        ret = 0
    else:
        ret = 1
    return ret

if __name__ == '__main__':
    pass
