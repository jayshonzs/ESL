'''
Created on 2014-8-7

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import simulate_data

def euclidean(x1, x2):
    return np.linalg.norm(x1-x2)

def k_means(X, k, distance=euclidean):
    centers = np.zeros((k, len(X[0])))
    min_x1, max_x1 = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    min_x2, max_x2 = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    
    #initialization
    for i in range(k):
        centers[i][0] = random.random()*(max_x1-min_x1)
        centers[i][1] = random.random()*(max_x2-min_x2)
        
    #print centers
    
    #run
    while True:
        dclusters = {}
        rcluster = []
        #step M
        for i in range(len(X)):
            item = X[i]
            min_dist = 999999999.
            center = 0
            for j in range(k):
                dist = distance(item, centers[j])
                if dist < min_dist:
                    min_dist = dist
                    center = j
            dclusters.setdefault(center, [])
            dclusters[center].append(i)
            rcluster.append(center)
        
        #step E
        ncenters = np.zeros((k, len(X[0])))
        for c in range(k):
            if c in dclusters:
                ncenters[c] = sum([X[idx] for idx in dclusters[c]])/len(dclusters[c])
        
        total_norm_diff = sum([np.linalg.norm(ncenters[i]-centers[i]) for i in range(k)])
        centers = ncenters
        if total_norm_diff < 0.000001:
            break    
    
    return rcluster, centers

def train(X, K=3, R=5):
    model = []
    for k in range(K):
        start_idx = 100*k
        rcluster, centers = k_means(X[start_idx:start_idx+100], R)
        print centers
        model.append((rcluster, centers))
    return model

def predict(model, x, distance=euclidean):
    min_distance = 999999999.
    best_k = None
    for k in range(len(model)):
        for r in range(len(model[k][1])):
            d = distance(x, model[k][1][r])
            if d < min_distance:
                min_distance = d
                best_k = k
    return best_k

def extract_centers(model):
    centers = []
    t = []
    for k in range(len(model)):
        for r in range(len(model[k][1])):
            center = model[k][1][r]
            centers.append(center)
            t.append(k)
    return np.array(centers), np.array(t)

def draw(data, classes, model, resolution=100):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = data[:, 0].min()-0.1, data[:, 0].max()+0.1
    two_min, two_max = data[:, 1].min()-0.1, data[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        z.append(predict(model, inputs[i]))
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    plt.scatter(data[:, 0], data[:, 1], s=50, c=classes, cmap=mycm)
    
    centers, t = extract_centers(model)
    plt.scatter(centers[:, 0], centers[:, 1], s=150, c=t, cmap=mycm)
    
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    
    plt.show()

if __name__ == '__main__':
    X = simulate_data.loaddata()
    t = []
    for i in range(300):
        if i < 100:
            t.append(0)
        elif i < 200:
            t.append(1)
        else:
            t.append(2)
    model = train(X)
    draw(X, np.array(t), model)
