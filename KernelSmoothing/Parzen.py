'''
Created on 2014-6-21

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
from LinearClassification import SAHeart_data as CHD

def cookdata(inputs, outputs):
    sbp_chd = []
    sbp_no_chd = []
    for i in range(len(inputs)):
        if outputs[i] == 1:
            sbp_chd.append(inputs[i, 0])
        else:
            sbp_no_chd.append(inputs[i, 0])
    return np.array(sbp_chd), np.array(sbp_no_chd)

def weight(x, xi, t):
    return np.exp(-(x-xi)*(x-xi)/(2*t))

def parzen(x0, X):
    s = 0.
    for i in range(len(X)):
        s += weight(x0, X[i], 30)
    return s/(len(X)*30)

def draw(sbp_chd, sbp_no_chd):
    predictors = np.arange(100., 220., 0.5)
    responses = np.zeros(len(predictors))
    for i in range(len(predictors)):
        responses[i] = parzen(predictors[i], sbp_chd)
    plt.plot(predictors, responses)
    for i in range(len(predictors)):
        responses[i] = parzen(predictors[i], sbp_no_chd)
    plt.plot(predictors, responses)

if __name__ == '__main__':
    inputs, outputs = CHD.loaddata()
    sbp_chd, sbp_no_chd = cookdata(inputs, outputs)
    draw(sbp_chd, sbp_no_chd)
    plt.show()
