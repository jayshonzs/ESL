'''
Created on 2014-4-28

@author: xiajie
'''
import numpy as np
import random
import matplotlib.pyplot as plt

def simulate():
    cov = [[1,0],[0,1]]
    cov2 = [[0.2,0],[0,0.2]]
    l = range(10)
    p1 = np.zeros((100,2))
    p2 = np.zeros((100,2))
    
    m = np.random.multivariate_normal((1,0),cov,10)
    n = np.random.multivariate_normal((0,1),cov,10)
    
    print m
    print n
    
    np.savetxt('scenters1.txt', m)
    np.savetxt('scenters2.txt', n)
    
    for i in range(100):
        idx = random.choice(l)
        p1[i] = np.random.multivariate_normal((m[idx][0],m[idx][1]),cov2,1)
        
        idx = random.choice(l)
        p2[i] = np.random.multivariate_normal((n[idx][0],n[idx][1]),cov2,1)
    
    return p1, p2

def draw_data(Data1, Data2):
    
    plt.scatter(Data1[:, 0], Data1[:, 1], s=50, c='r')
    plt.scatter(Data2[:, 0], Data2[:, 1], s=50, c='g')
    
    plt.show()

if __name__ == '__main__':
    data1, data2 = simulate()
    np.savetxt('sdata1.txt', data1)
    np.savetxt('sdata2.txt', data2)
    draw_data(data1, data2)
