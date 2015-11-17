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

def ave(a):
    return sum(a)/len(a)

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

def quadratic(X, Y, col, val):
    X1, X2, Y1, Y2 = divideset(X, Y, col, val)
    if len(Y1) == 0 or len(Y2) == 0:
        return None
    c1 = ave(Y1)
    c2 = ave(Y2)
    q1 = 0.
    q2 = 0.
    for i in range(len(Y1)):
        q1 += (Y1[i]-c1)**2
    for i in range(len(Y2)):
        q2 += (Y2[i]-c2)**2
    return (q1+q2, X1, X2, Y1, Y2)

def buildrandomtree(X, Y, m, depth=0, max_depth=2):
    if len(X) == 0:
        return None
    elif len(X) <= 5:
        return decisionnode(result=ave(Y), sets=(X,Y))
    
    min_quadratic = 999999999999.0
    best_criteria = None
    best_sets = None
    
    rd_cols = [i for i in range(len(X[0]))]
    rd_cols = random.shuffle(rd_cols)[:m]
    rd_cols.sort()
    
    for col in rd_cols:
        column = [X[i][col] for i in range(len(X))]
        for s in range(len(column)):
            res = quadratic(X, Y, col, column[s])
            if res == None:
                continue
            q, X1, X2, Y1, Y2 = res
            c = ave(Y)
            total_q = 0.
            for i in range(len(Y)):
                total_q += (Y[i]-c)**2
            if q < total_q and q < min_quadratic:
                min_quadratic = q
                best_criteria = (col, column[s])
                best_sets = [(X1, Y1),(X2, Y2)]
    if len(best_sets[0][0]) > 0 and len(best_sets[1][0]) > 0 and depth < max_depth:
        leftBranch = buildrandomtree(best_sets[0][0], best_sets[0][1], m, depth+1, max_depth)
        rightBranch = buildrandomtree(best_sets[1][0], best_sets[1][1], m, depth+1, max_depth)
        return decisionnode(col=best_criteria[0], value=best_criteria[1], left_child=leftBranch, right_child=rightBranch, sets=(X,Y))
    else:
        return decisionnode(result=ave(Y), sets=(X,Y))

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
    B = len(forest)
    res = 0.
    for tree in forest:
        res += tree_predict(obs, tree)
    return res/B

if __name__ == '__main__':
    pass
