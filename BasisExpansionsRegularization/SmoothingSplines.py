'''
Created on 2014-7-8

@author: xiajie
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad

def convertfunc(x):
    if x == 'male':
        return 0.
    else:
        return 1.

def load():
    inputs = np.genfromtxt('bone.data', delimiter='\t', converters={2:convertfunc}, skip_header=1, dtype=float, usecols=(1,2))
    outputs = np.genfromtxt('bone.data', delimiter='\t', skip_header=1, dtype=float, usecols=(3))
    return inputs, outputs

def cookdata(inputs, outputs):
    male_inputs = []
    female_inputs = []
    male_outputs = []
    female_outputs = []
    for i in range(len(inputs)):
        if inputs[i][1] == 0.:
            male_inputs.append(inputs[i][0])
            male_outputs.append(outputs[i])
        else:
            female_inputs.append(inputs[i][0])
            female_outputs.append(outputs[i])
    return male_inputs, male_outputs, female_inputs, female_outputs

def d(x, k, knots):
    kdiff = knots[-1] - knots[k-1]
    item1 = x-knots[k-1]
    if item1 <= 0:
        item1 = 0
    item2 = x-knots[-1]
    if item2 <= 0:
        item2 = 0
    if kdiff == 0:
        return 0
    return (item1**3 - item2**3)/kdiff

def N(x, k, knots):
    k += 1
    if k == 1:
        return 1
    elif k == 2:
        return x
    else:
        return d(x, k-2, knots)-d(x, len(knots)-1, knots)

def d2bar(x, k, knots):
    kdiff = knots[-1] - knots[k-1]
    item1 = x-knots[k-1]
    if item1 <= 0:
        item1 = 0
    item2 = x-knots[-1]
    if item2 <= 0:
        item2 = 0
    if kdiff == 0:
        return 0
    return 6*(item1 - item2)/kdiff

def N2bar(x, k, knots):
    k += 1
    if k == 1:
        return 0
    elif k == 2:
        return 0
    else:
        return d(x, k-2, knots)-d(x, len(knots)-1, knots)

def mul(x, i, j, knots):
    return N2bar(x, i, knots)*N2bar(x, j, knots)
    
def integrate(i, j, knots, xmin, xmax):
    return quad(mul, xmin, xmax, args=(i,j, knots))

def omega(knots, xmin, xmax):
    length = len(knots)
    omg = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            omg[i,j] = integrate(i, j, knots, xmin, xmax)[0]-integrate(i, j, knots, xmin, xmax)[1]
    return omg
            
def theta(Y, omega, N, lmbda=0.00022):
    NT = np.transpose(N)
    M = NT.dot(N)+omega*lmbda
    print np.linalg.matrix_rank(omega)
    print M.shape, np.linalg.matrix_rank(M)
    return np.linalg.inv(M).dot(NT)*Y

def draw(inputs, outputs, male_theta, female_theta, male_knots, female_knots, resolution=50):
    mycm = mpl.cm.get_cmap('Paired')
    
    minx = inputs[:,0].min()
    maxx = inputs[:,0].max()
    X = np.arange(minx, maxx, 100)
    male_N = []
    for k in range(len(X)):
        male_N.append(N(X[k], k, male_knots))
    female_N = []
    for k in range(len(X)):
        female_N.append(N(X[k], k, female_knots))
    male_N_array = np.array(male_N)
    female_N_array = np.array(female_N)
    
    male_Y = male_N_array.dot(male_theta)
    female_Y = female_N_array.dot(female_theta)
    
    plt.scatter(inputs[:, 0], outputs, s=50, c=inputs[:,1], cmap=mycm)
    plt.plot(X, male_Y)
    plt.plot(X, female_Y)
    
    plt.show()

if __name__ == '__main__':
    inputs, outputs = load()
    male_inputs, male_outputs, female_inputs, female_outputs = cookdata(inputs, outputs)
    male_knots = male_inputs
    female_knots = female_inputs
    male_min = sorted(male_knots)[0]
    male_max = sorted(male_knots)[-1]
    female_min = sorted(female_knots)[0]
    female_max = sorted(female_knots)[-1]
    male_N = []
    for k in range(len(male_inputs)):
        male_N.append(N(male_inputs[k], k, male_knots))
    female_N = []
    for k in range(len(female_inputs)):
        female_N.append(N(female_inputs[k], k, female_knots))
    male_N_array = np.array(male_N)
    female_N_array = np.array(female_N)
    #print male_N_array
    #print female_N_array
    print male_min, male_max, female_min, female_max
    #male_omg = omega(male_knots, male_min, male_max)
    #female_omg = omega(female_knots, female_min, female_max)
    male_omg = np.genfromtxt('male_omg.data', dtype=float)
    female_omg = np.genfromtxt('female_omg.data', dtype=float)
    print male_omg.shape
    print female_omg.shape
    #np.savetxt('male_omg.data', male_omg)
    #np.savetxt('female_omg.data', female_omg)
    male_theta = theta(np.array(male_outputs), male_omg, male_N_array)
    female_theta = theta(np.array(female_outputs), female_omg, female_N_array)
    print male_theta
    print female_theta
    draw(inputs, outputs, male_theta, female_theta, male_knots, female_knots)
    