'''
Created on 2014-6-7

@author: xiajie
'''
import numpy as np
import vowel_data as vw
import matplotlib.pyplot as plt
import matplotlib as mpl

def cookdata(inputs, outputs):
    classes = {}
    for i in range(len(outputs)):
        classes.setdefault(outputs[i],[])
        classes[outputs[i]].append(inputs[i].tolist())
    for key in classes.keys():
        classes[key] = np.array(classes[key])
    return classes

def centroids(data, K, p):
    M = np.zeros((K,p))
    for i in range(len(data)):
        M[i] = np.mean(data[i+1],axis=0)
    return M

def withinclasscov(data, M, p):
    W = np.zeros((p,p))
    for key in data.keys():
        idx = key-1
        means = M[idx]
        g = np.zeros((p,p))
        for i in range(len(data[key])):
            diff = data[key][i]-means
            g += np.outer(diff, diff)
        g = g/(len(data[key])-1)
        W += g
    
    W /= len(data)
    
    return W

def classcov(M, p):
    B = np.zeros((p,p))
    
    means = np.mean(M, axis=0)
    for i in range(len(M)):
        diff = M[i]-means
        B += np.outer(diff, diff)
    
    B = B/(len(M)-1)
    
    return B

def eigenvectors(M, W):
    w, v = np.linalg.eig(W)
    diag = np.diag(w**(-0.5))
    W_half = v.dot(diag).dot(np.transpose(v))
    M_star = M.dot(W_half)
    B_star = classcov(M_star,len(M[0]))
    bw, bv = np.linalg.eig(B_star)
    
    for i in range(len(bv[0])):
        bv[:,i] = W_half.dot(bv[:,i])
    
    return bv

def reducedata(data, v):
    ret = []
    ret_type = []
    for i in data.keys():
        for j in range(len(data[i])):
            local0 = v[:,0].dot(data[i][j])
            local1 = v[:,1].dot(data[i][j])
            ret.append([local0,local1])
            ret_type.append(i)
    return np.array(ret), np.array(ret_type)

def reducecentroids(M, v):
    ret = []
    for i in range(len(M)):
        local0 = v[:,0].dot(M[i])
        local1 = v[:,1].dot(M[i])
        ret.append([local0, local1])
    return np.array(ret)

def classify(M, x):
    dmin = 999999999
    index = 0
    for i in range(len(M)):
        diff = M[i]-x
        #print i, M[i], x, diff
        sdist = np.sum(diff**2)
        if sdist < dmin:
            dmin = sdist
            index = i
    #print index+1
    return int(index+1)

def draw(data, outputs, centroids, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = data[:, 0].min()-0.1, data[:, 0].max()+0.1
    two_min, two_max = data[:, 1].min()-0.1, data[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        z.append(classify(centroids, inputs[i]))
    result = np.array(z).reshape(xx1.shape)
    np.savetxt('result.out', result, delimiter=',', fmt='%2d')
    plt.contourf(xx1, xx2, result, 12, cmap=mycm)
    
    plt.scatter(data[:, 0], data[:, 1], s=50, c=outputs, cmap=mycm)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array([1,2,3,4,5,6,7,8,9,10,11]), marker='^', s=200, cmap=mycm)
    
    plt.show()
    
def test(inputs, outputs, v):
    classes = cookdata(inputs, outputs)
    M = centroids(classes, len(classes), len(inputs[0]))
    reduced, reduces_type = reducedata(classes, v)
    reduced_cent = reducecentroids(M, v)
    
    correct = 0.
    for i in range(len(reduced)):
        ret = classify(reduced_cent, reduced[i])
        print ret, reduces_type[i]
        if ret == reduces_type[i]:
            correct += 1.
    
    print 'error rate:', 1-correct/len(reduced)

if __name__ == '__main__':
    train_input,train_output,test_input,test_output = vw.loaddata()
    classes = cookdata(train_input, train_output)
    M = centroids(classes, len(classes), len(train_input[0]))
    W = withinclasscov(classes, M, len(train_input[0]))
    v = eigenvectors(M, W)
    reduced, reduces_type = reducedata(classes, v)
    reduced_cent = reducecentroids(M, v)
    draw(reduced, reduces_type, reduced_cent, 200.)
    test(test_input, test_output, v)