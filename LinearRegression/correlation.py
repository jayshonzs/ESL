'''
Created on 2014-5-4

@author: xiajie
'''
import numpy as np

def correlation(X, Y):
    EXY = np.mean(X*Y)
    EX2 = np.mean(X**2)
    EY2 = np.mean(Y**2)
    EX = np.mean(X)
    EY = np.mean(Y)
    
    return (EXY-EX*EY)/(np.sqrt(EX2-EX**2)*np.sqrt(EY2-EY**2))
