'''
Created on 2014-7-12

@author: xiajie
'''
import copy
from LinearRegression import prostate

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

def buildtree(X, Y, depth=0, max_depth=99999999):
    if len(X) == 0:
        return None
    elif len(X) <= 5:
        return decisionnode(result=ave(Y), sets=(X,Y))
    
    min_quadratic = 999999999999.0
    best_criteria = None
    best_sets = None
    
    column_count = len(X[0])
    for col in range(column_count):
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
        leftBranch = buildtree(best_sets[0][0], best_sets[0][1], depth+1, max_depth)
        rightBranch = buildtree(best_sets[1][0], best_sets[1][1], depth+1, max_depth)
        return decisionnode(col=best_criteria[0], value=best_criteria[1], left_child=leftBranch, right_child=rightBranch, sets=(X,Y))
    else:
        return decisionnode(result=ave(Y), sets=(X,Y))

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

def cost_complexity(tree_node, total_q=0., terminal_num=0):
    if tree_node == None:
        return 0,0
    if tree_node.left_child == None and tree_node.right_child == None:
        terminal_num += 1
        q = 0.
        sets = tree_node.sets
        c = ave(sets[1])
        for i in range(len(sets[1])):
            q += (sets[1][i]-c)**2
        total_q += q
    else:
        total_q, terminal_num = cost_complexity(tree_node.left_child, total_q, terminal_num)
        total_q, terminal_num = cost_complexity(tree_node.right_child, total_q, terminal_num)
    return total_q, terminal_num

def do_prune(tree, t):
    if tree == None:
        return False
    node = tree
    if node.equal(t):
        node.col = None
        node.value = None
        node.result = ave(node.sets[1])
        node.left_child = None
        node.right_child = None
        return True
    else:
        if do_prune(node.left_child, t) == True:
            return True
        elif do_prune(node.right_child, t) == True:
            return True    
    return False

def possible_pruning(tree, node, tree_list):
    #internal node
    if node.result == None:
        new_tree = copy.deepcopy(tree)
        if do_prune(new_tree, node) == False:
            print 'pruning error.'
        tree_list.append(new_tree)
        possible_pruning(tree, node.left_child, tree_list)
        possible_pruning(tree, node.right_child, tree_list)

def pruning(tree):
    tree_list = []
    orig_cost = 0.
    orig_terms = 0
    orig_cost, orig_terms = cost_complexity(tree, orig_cost, orig_terms)
    print ':', orig_cost, orig_terms
    possible_pruning(tree, tree, tree_list)
    dmin = 999999999.
    best_tree = None
    for sub_tree in tree_list:
        cost = 0.
        terms = 0
        cost, terms = cost_complexity(sub_tree, cost, terms)
        print cost, terms
        diff = (cost-orig_cost)/(orig_terms-terms)
        if diff < dmin:
            dmin = diff
            best_tree = sub_tree
    return best_tree

def selection(tree, alpha=0.1):
    node = tree
    sub_trees = []
    while node.result == None:
        sub_tree = pruning(node)
        node = sub_tree
        sub_trees.append(sub_tree)
    mincc = 999999999.
    best_tree = None
    print len(sub_trees)
    for sub_tree in sub_trees:
        cost = 0.
        terminals = 0
        cost, terminals = cost_complexity(sub_tree, cost, terminals)
        cc = cost + alpha*terminals
        if cc < mincc:
            mincc = cc
            best_tree = sub_tree
    return best_tree

def cookdata(inputs, outputs, Ttype):
    dlist = inputs.tolist()
    train_cooked = []
    test_cooked = []
    train_output = []
    test_output = []
    for i in range(len(dlist)):
        if Ttype[i] == 'T':
            train_cooked.append(dlist[i])
            train_output.append(outputs[i])
        else:
            test_cooked.append(dlist[i])
            test_output.append(outputs[i])
    return train_cooked,train_output,test_cooked,test_output

if __name__ == '__main__':
    inputs, outputs, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = cookdata(inputs, outputs, Ttype)
    tree = buildtree(train_data, train_out)
    node = tree
    pruned_tree = selection(tree, 1)
    total = 0.
    for i in range(len(test_data)):
        res = predict(test_data[i], pruned_tree)
        print res, test_out[i]
        total += (res-test_out[i])**2
    print total/len(test_out)
