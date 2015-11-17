'''
Created on 2014-5-26

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def loaddata():
    train_input = np.genfromtxt('vowel.train', delimiter=',', skip_header=1, dtype=float, usecols=(2,3,4,5,6,7,8,9,10,11))
    train_output = np.genfromtxt('vowel.train', delimiter=',', skip_header=1, dtype=float, usecols=(1))
    test_input = np.genfromtxt('vowel.test', delimiter=',', skip_header=1, dtype=float, usecols=(2,3,4,5,6,7,8,9,10,11))
    test_output = np.genfromtxt('vowel.test', delimiter=',', skip_header=1, dtype=float, usecols=(1))
    return train_input, train_output, test_input, test_output

def draw(inputs, outputs):
    mycm = mpl.cm.get_cmap('Paired')
    
    plt.scatter(inputs[:, 0], inputs[:, 1], s=50, c=outputs, cmap=mycm)
    
    plt.show()

if __name__ == '__main__':
    train_input,train_output,test_input,test_output = loaddata()
    draw(train_input, train_output)
    print train_input.shape
    print train_output
    