'''
Created on 2014-8-7

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import simulate_data
import K_means

def euclidean(x1, x2):
    return np.linalg.norm(x1-x2)

def move(center, x, eps):
    d = x - center
    center = center + eps*d

def train(X, model, distance=euclidean):
    eps = 0.05
    while eps > 0.00001:
        for i, x in enumerate(X):
            c = None
            if i < 100:
                c = 0
            elif i < 200:
                c = 1
            else:
                c = 2
            k, t = predict(model, x)
            if k == c:
                move(model[t], x, eps)
            else:
                move(model[t], x, -eps)
        eps *= 0.95
    return model

def predict(model, x, distance=euclidean):
    min_distance = 999999999.
    best_i = None
    best_k = None
    for i in range(len(model)):
        d = distance(x, model[i])
        if d < min_distance:
            min_distance = d
            best_i = i
    if best_i < 5:
        best_k = 0
    elif best_i < 10:
        best_k = 1
    else:
        best_k = 2
    return best_k, best_i

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
        z.append(predict(model, inputs[i])[0])
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    plt.scatter(data[:, 0], data[:, 1], s=50, c=classes, cmap=mycm)
    
    t = np.zeros(15)
    for i in range(15):
        if i < 5:
            t[i] = 0
        elif i < 10:
            t[i] = 1
        else:
            t[i] = 2
    plt.scatter(model[:, 0], model[:, 1], s=150, c=t, cmap=mycm)
    
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
    k_means_model = K_means.train(X)
    init_model = extract_centers(k_means_model)[0]
    model = train(X, init_model)
    draw(X, np.array(t), model)
