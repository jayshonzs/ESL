'''
Created on 2014-7-23

@author: xiajie
'''
import numpy as np
from AdditiveModelsTreesRelatedModels import RegressionTree
from LinearRegression import prostate
from PIL import Image, ImageDraw

def gradient_quad(Model, x, y):
    f = predict(Model, x)
    #print f, y
    return y-f

def fit(Model, tree):
    tree_node = tree
    if tree_node == None:
        return
    #print 'start fiting...'
    #drawtree(tree)
    while True:
        #print tree_node.left_child, tree_node.right_child
        if tree_node.result != None:
            X = tree.sets[0]
            Y = tree.sets[1]
            N = len(Y)
            residual = np.zeros(N)
            for i in range(N):
                res = predict(Model, X[i])
                residual[i] = Y[i]-res
            tree_node.result = np.mean(residual)
            #print "result:", tree_node.result
            return
        else:
            #print 'left'
            fit(Model, tree_node.left_child)
            #print 'right'
            fit(Model, tree_node.right_child)
            return

def gbm(X, Y, M=50, gradient=gradient_quad):
    N = len(Y)
    Model = [np.mean(Y)]
    for m in range(M):
        print '%d:%f' % (m, RSS(Model, X, Y))
        residual = np.zeros(N)
        for i in range(N):
            residual[i] = gradient(Model, X[i], Y[i])
        #print residual
        new_tree = RegressionTree.buildtree(X, residual, max_depth=1)
        print 'New tree builded.'
        #fit(Model, new_tree)
        print 'Tree fited.'
        Model.append(new_tree)
    print '%d:%f' % (m, RSS(Model, X, Y))
    return Model

def predict(Model, x, v=0.05):
    f = Model[0]
    for model in Model[1:]:
        res = RegressionTree.predict(x, model)
        f += res*v
    return f

def RSS(Model, X, Y):
    N = len(Y)
    err = 0.
    for i in range(N):
        res = predict(Model, X[i])
        err += (res-Y[i])**2
    return err/len(Y)

def node_rss(tree_node):
    X = tree_node.sets[0]
    Y = tree_node.sets[1]
    result = np.mean(Y)
    err = 0.
    for i in range(len(X)):
        err += (result-Y[i])**2
    return err

def importances(tree, P, im={}):
    node = tree
    if node == None:
        return
    if node.left_child != None and node.right_child != None:
        col = node.col
        im.setdefault(col, 0.)
        total_rss = node_rss(node)
        left_rss = node_rss(node.left_child)
        right_rss = node_rss(node.right_child)
        rss_im = total_rss-left_rss-right_rss
        im[col] += rss_im
        importances(node.left_child, P, im)
        importances(node.right_child, P, im)
    return

def relative_importance(Model, P):
    m = len(Model)-1
    total = {}
    for model in Model[1:]:
        im = {}
        importances(model, P, im)
        for key in im.keys():
            total.setdefault(key, 0.)
            total[key] += im[key]
    for key in total.keys():
        total[key] /= m
    return total

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
    inputs, output, Ttype = prostate.loaddata()
    train_data, train_out, test_data, test_out = RegressionTree.cookdata(inputs, output, Ttype)
    P = len(train_data[0])
    print 'Training...'
    Model = gbm(train_data, train_out)
    print 'Trained!'
    total = 0.
    for i in range(len(test_data)):
        res = predict(Model, test_data[i])
        print res, test_out[i]
        total += (res-test_out[i])**2
    print total/len(test_out)
    
    print '**************************'
    im = relative_importance(Model, P)
    for key in im.keys():
        print key, im[key]
