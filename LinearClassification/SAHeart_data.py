'''
Created on 2014-6-7

@author: xiajie
'''
import numpy as np

def convertfunc(x):
    if x == 'Present':
        return 1.
    else:
        return 0.

def loaddata():
    inputs = np.genfromtxt('SAheart.data', delimiter=',', converters={5:convertfunc}, skip_header=1, dtype=float, usecols=(1,2,3,5,7,8,9))
    outputs = np.genfromtxt('SAheart.data', delimiter=',', skip_header=1, dtype=float, usecols=(10))
    return inputs, outputs

if __name__ == '__main__':
    inputs, outputs = loaddata()
    print inputs
