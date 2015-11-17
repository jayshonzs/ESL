'''
Created on 2014-7-15

@author: xiajie
'''
import numpy as np
import spam_data as spam
import copy
from PIL import Image, ImageDraw

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

def buildtree(X, Y):
    if len(X) == 0:
        return None
    elif len(X) <= 5:
        return decisionnode(result=clss(Y), sets=(X,Y))
    
    min_deviance = 999999999999.0
    best_criteria = None
    best_sets = None
    
    column_count = len(X[0])
    for col in range(column_count):
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
        leftBranch = buildtree(best_sets[0][0], best_sets[0][1])
        rightBranch = buildtree(best_sets[1][0], best_sets[1][1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1], left_child=leftBranch, right_child=rightBranch, sets=(X,Y))
    else:
        return decisionnode(result=clss(Y), sets=(X,Y))

def selection(tree, alpha=1.):
    print "start selection."
    node = tree
    sub_trees = []
    while node.result == None:
        sub_tree, terms = pruning(node)
        node = sub_tree
        sub_trees.append(sub_tree)
        orig_cost = 0.
        orig_terms = 0
        orig_cost, orig_terms = cost_complexity(node, orig_cost, orig_terms)
        print "after pruning:", orig_cost, orig_terms, terms
    mincc = 999999999.
    best_tree = None
    best_terminals = None
    print "subtrees: ", len(sub_trees)
    for sub_tree in sub_trees:
        cost = 0.
        terminals = 0
        cost, terminals = cost_complexity(sub_tree, cost, terminals)
        print "selection:", cost, terminals
        cc = cost + alpha*terminals
        if cc < mincc:
            mincc = cc
            best_tree = sub_tree
            best_terminals = terminals
    return best_tree, best_terminals

def pruning(tree):
    tree_list = []
    orig_cost = 0.
    orig_terms = 0
    orig_cost, orig_terms = cost_complexity(tree, orig_cost, orig_terms)
    print 'pruning orig:', orig_cost, orig_terms
    possible_pruning(tree, tree, tree_list)
    dmin = 999999999.
    best_tree = None
    for sub_tree in tree_list:
        cost = 0.
        terms = 0
        cost, terms = cost_complexity(sub_tree, cost, terms)
        diff = (cost-orig_cost)/(orig_terms-terms)
        if diff < dmin:
            dmin = diff
            best_tree = sub_tree
    return best_tree, terms

def possible_pruning(tree, node, tree_list):
    #internal node
    if node.result == None:
        new_tree = copy.deepcopy(tree)
        if do_prune(new_tree, node) == False:
            print 'pruning error.'
        tree_list.append(new_tree)
        possible_pruning(tree, node.left_child, tree_list)
        possible_pruning(tree, node.right_child, tree_list)

def do_prune(tree, t):
    if tree == None:
        return False
    node = tree
    if node.equal(t):
        node.col = None
        node.value = None
        node.result = clss(node.sets[1])
        node.left_child = None
        node.right_child = None
        return True
    else:
        if do_prune(node.left_child, t) == True:
            return True
        elif do_prune(node.right_child, t) == True:
            return True    
    return False

def cost_complexity(tree_node, total_q, terminal_num):
    if tree_node == None:
        return 0,0
    if tree_node.left_child == None and tree_node.right_child == None:
        terminal_num += 1
        q = 0.
        sets = tree_node.sets
        q = missclassifyerror(tree_node)*len(sets[0])
        total_q += q
    else:
        total_q, terminal_num = cost_complexity(tree_node.left_child, total_q, terminal_num)
        total_q, terminal_num = cost_complexity(tree_node.right_child, total_q, terminal_num)
    return total_q, terminal_num

def predict(obs, tree):
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

def getwidth(tree):
    if tree.left_child == None and tree.right_child == None:
        return 1
    return getwidth(tree.left_child) + getwidth(tree.right_child)

def getdepth(tree):
    if tree.left_child == None and tree.right_child == None:
        return 0
    return max(getdepth(tree.left_child), getdepth(tree.right_child))+1

def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree)*100
    h = getdepth(tree)*100 + 120
    
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    drawnode(draw, tree, w/2, 20)
    img.save(jpeg, 'JPEG')

def drawnode(draw, tree, x, y):
    if tree.result == None:
        w1 = getwidth(tree.left_child)*100
        w2 = getwidth(tree.right_child)*100
        
        left = x-(w1+w2)/2
        right = x+(w1+w2)/2
        
        draw.text((x-20,y-10), str(tree.col)+':'+str(tree.value), (0, 0, 0))
        
        draw.line((x, y, left+w1/2, y+100), fill=(255, 0, 0))
        draw.line((x, y, right-w2/2, y+100), fill=(255, 0, 0))
        
        drawnode(draw, tree.left_child, left+w1/2, y+100)
        drawnode(draw, tree.right_child, right-w2/2, y+100)
    else:
        txt = '%d' % tree.result
        draw.text((x-20, y), txt, (0, 0, 0))

if __name__ == '__main__':
    X_test, X_train, Y_test, Y_train = spam.load()
    #X_train = X_train[:500] + X_train[-500:]
    #Y_train = Y_train[:500] + Y_train[-500:]
    tree = buildtree(X_train, Y_train)
    pruned_tree, terminals = selection(tree, 1)
    drawtree(pruned_tree)
    print terminals
    total = 0.
    for i in range(len(X_test)):
        res = predict(X_test[i], pruned_tree)
        #print res, Y_test[i]
        if res != Y_test[i]:
            total += 1
    print total/len(Y_test)
