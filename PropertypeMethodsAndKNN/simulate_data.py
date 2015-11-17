'''
Created on 2014-8-7

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

def loaddata(dfile='gaussian_mix.dat'):
    data = np.genfromtxt(file(dfile))
    #print data
    return data

def savedata(data, dfile='gaussian_mix.dat'):
    np.savetxt(dfile, data)
    #print data
    return data

def fix_propertypes(K=3, R=5):
    centers = np.zeros((K*R, 3))
    centers[0] = np.array([1.5, 8.5, 0])
    centers[1] = np.array([2.3, 6.5, 0])
    centers[2] = np.array([3., 5., 0])
    centers[3] = np.array([5.5, 7.5, 0])
    centers[4] = np.array([7., 5.5, 0])
    
    centers[5] = np.array([2., 3.5, 1])
    centers[6] = np.array([3., 1.5, 1])
    centers[7] = np.array([3.5, 4., 1])
    centers[8] = np.array([7., 3., 1])
    centers[9] = np.array([8., 1., 1])
    
    centers[10] = np.array([3.5, 5., 2])
    centers[11] = np.array([4.5, 6., 2])
    centers[12] = np.array([5., 2., 2])
    centers[13] = np.array([6., 4.5, 2])
    centers[14] = np.array([8., 6.5, 2])
    
    return centers
    
def simulate(K=3, R=5):
    data_set = []
    centers = fix_propertypes(K, R)
    cov = [[1.5, 0.], [0., 1.5]]
    for c in range(len(centers)):
        data = np.zeros((20, 2))
        for i in range(20):
            x = np.random.multivariate_normal(centers[c][:2], cov, 1)
            data[i] = x
        data_set.append(data)
    return data_set

def draw_clusters(centroids, X, clusters):
    mycm = mpl.cm.get_cmap('Paired')
    
    plt.scatter(X[:, 0], X[:, 1], s=30, c=clusters, cmap=mycm)
    
    #plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c=centroids[:, 2], cmap=mycm)
    
    #plt.xlim([0, 10])
    #plt.ylim([0, 10])
    
    plt.show()

def merge_data(data):
    ret = np.zeros((300, 2))
    i = 0
    for r in range(len(data)):
        for j in range(len(data[r])):
            ret[i] = data[r][j]
            i += 1
    return ret

def shuffle_data():
    data = loaddata('x.dat')
    lst = data.tolist()
    random.shuffle(lst)
    savedata(np.array(lst[:300]),'shuffled.dat')
    return 0

def draw_mpi_result():
    X = np.zeros((300, 2))
    clusters = np.zeros(300)
    data = np.genfromtxt(file('result.dat'))
    for i in range(300):
        X[i] = data[i, :2]
        clusters[i] = data[i, 2]
    draw_clusters(None, X, clusters)
    return

if __name__ != '__main__':
    centers = fix_propertypes()
    data = simulate()
    X = merge_data(data)
    print X
    print X.shape
    t = []
    for i in range(300):
        if i < 100:
            t.append(0)
        elif i < 200:
            t.append(1)
        else:
            t.append(2)
    savedata(X)
    LX = loaddata()
    draw_clusters(centers, LX, np.array(t))
else:
    #shuffle_data()
    draw_mpi_result()
