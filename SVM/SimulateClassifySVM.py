'''
Created on 2014-8-5

@author: xiajie
'''
import numpy as np
import SMO
import matplotlib.pyplot as plt
import matplotlib as mpl

def simulate_data():
    data_set = []
    centers = [(5.,1.),(3.,2.)]
    cov = [[[0.5,.2],[.2,0.5]],[[2.,0.],[0.,2.]]]
    for k in range(len(centers)):
        center = centers[k]
        data = np.zeros((50, 2))
        for i in range(50):
            x = np.random.multivariate_normal(center, cov[k], 1)
            data[i] = x
        data_set.append(data)
    return data_set, 100

def cook_data(data_set):
    data = np.zeros((len(data_set)*len(data_set[0]),2))
    classes = np.zeros(len(data))
    idx = 0
    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            data[idx] = data_set[i][j]
            if i == 0:
                classes[idx] = -1
            else:
                classes[idx] = 1
            idx += 1
    return data, classes

def draw(data, classes, sv, alphas, b, resolution):
    mycm = mpl.cm.get_cmap('Paired')
    
    one_min, one_max = data[:, 0].min()-0.1, data[:, 0].max()+0.1
    two_min, two_max = data[:, 1].min()-0.1, data[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(one_min, one_max, (one_max-one_min)/resolution),
                     np.arange(two_min, two_max, (two_max-two_min)/resolution))
    
    inputs = np.c_[xx1.ravel(), xx2.ravel()]
    z = []
    for i in range(len(inputs)):
        z.append(SMO.predict(data, classes, alphas, b, inputs[i]))
    result = np.array(z).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, result, cmap=mycm)
    plt.scatter(data[:, 0], data[:, 1], s=50, c=classes, cmap=mycm)
    plt.scatter(sv[:,0], sv[:,1], s=10)
    
    plt.show()

if __name__ == '__main__':
    data_set, N = simulate_data()
    data, classes = cook_data(data_set)
    #print data
    #print classes
    alphas, b = SMO.run(data, classes)
    print alphas
    print b
    sv = []
    for i in range(len(alphas)):
        if alphas[i] >= 0.001 and alphas[i] <= 100.0:
            print alphas[i], data[i]
            sv.append(data[i])
    sv = np.array(sv)
    print sv
    draw(data, classes, sv, alphas, b, 100.)
