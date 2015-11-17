'''
Created on 2014-8-1

@author: xiajie
'''
import numpy as np

def fmax(a, b):
    if a >= b:
        return a
    else:
        return b
    
def fmin(a, b):
    if a <= b:
        return a
    else:
        return b
    
def radia_kernel(x1, x2):
    return np.transpose(x1).dot(x2)

def kernel(x1, x2):
    d = x1 - x2
    res = np.sum(d**2)
    return np.exp(-res)

def f(X, Y, alphas, x, b):
    N = len(alphas)
    ret = -b
    for i in range(N):
        if alphas[i] >= 0 and alphas[i] < 0.000001:
            continue
        if alphas[i] <= 0 and alphas[i] > -0.000001:
            continue
        ret += alphas[i]*Y[i]*radia_kernel(x,X[i])
    return ret

def W(X, Y, alphas, i, v):
    print 'WWWWWWW'
    tmp = alphas[i]
    alphas[i] = v
    N = len(Y)
    w = np.sum(alphas)
    s = 0.
    for i in range(N):
        for j in range(N):
            s += Y[i]*Y[j]*radia_kernel(X[i],X[j])*alphas[i]*alphas[j]
    w = w - 0.5*s
    alphas[i] = tmp
    return w

def takestep(Y, X, alphas, i1, i2, b, E, C=10):
    N = len(alphas)
    if i1 == i2:
        return 0
    alpha1 = alphas[i1]
    alpha2 = alphas[i2]
    y1 = Y[i1]
    y2 = Y[i2]
    x1 = X[i1]
    x2 = X[i2]
    if alphas[i1] > 0 and alphas[i1] < C:
        E1 = E[i1]
    else:
        E1 = f(X, Y, alphas, x1, b[0])-y1
    if alphas[i2] > 0 and alphas[i2] < C:
        E2 = E[i2]
    else:
        E2 = f(X, Y, alphas, x2, b[0])-y2
    s = y1*y2
    if y1 != y2:
        L = fmax(0, alpha2-alpha1)
        H = fmin(C, C+alpha2-alpha1)
    else:
        L = fmax(0, alpha1+alpha2-C)
        H = fmin(C, alpha1+alpha2)
    if L == H:
        return 0
    k11 = radia_kernel(x1, x1)
    k12 = radia_kernel(x1, x2)
    k22 = radia_kernel(x2, x2)
    eta = 2*k12-k11-k22
    eps = 0.001
    if eta < 0:
        a2 = alpha2-y2*(E1-E2)/eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        Lobj = W(X, Y, alphas, i2, L)
        Hobj = W(X, Y, alphas, i2, H)
        if Lobj > Hobj + eps:
            a2 = Lobj
        elif Lobj < Hobj - eps:
            a2 = Hobj
        else:
            a2 = alpha2
    if a2 < 1e-8:
        a2 = 0
    elif a2 > C-1e-8:
        a2 = C
    if abs(a2-alpha2) < eps*(a2+alpha2+eps):
        return 0
    a1 = alpha1 + s*(alpha2-a2)
    if a1 < 1e-8:
        a1 = 0
    elif a1 > C-1e-8:
        a1 = C
    
    b1 = E1 + y1*(a1-alpha1)*radia_kernel(x1,x1) + y2*(a2-alpha2)*radia_kernel(x1,x2) + b[0]
    b2 = E2 + y1*(a1-alpha1)*radia_kernel(x1,x2) + y2*(a2-alpha2)*radia_kernel(x2,x2) + b[0]
    if a1 == 0 or a1 == C:
        if a2 == 0 or a2 == C:
            new_b = (b1+b2)/2.
        else:
            new_b = b2
    else:
        new_b = b1

    for k in range(N):
        if alphas[k] > 0 and alphas[k] < C:
            if k == i1 or k == i2:
                E[k] = 0.
            else:
                E[k] = E[k] + y1*(a1-alpha1)*radia_kernel(x1,X[k]) + y2*(a2-alpha2)*radia_kernel(x2,X[k]) + b[0] - new_b
    
    alphas[i1] = a1
    alphas[i2] = a2
    b[0] = new_b
    print 'new_b:', new_b
    
    return 1

def secondheuristic(alphas, E, E1, i2, C):
    N = len(E)
    best_i = None
    if E1 >= 0:
        min_e = 999999999.
        for i in range(N):
            if i != i2 and alphas[i] > 0 and alphas[i] < C:
                if E[i] < min_e:
                    min_e = E[i]
                    best_i = i
    else:
        max_e = -999999999.
        for i in range(N):
            if i != i2 and alphas[i] > 0 and alphas[i] < C:
                if E[i] > max_e:
                    max_e = E[i]
                    best_i = i
    return best_i

def examineExample(X, Y, alphas, b, E, i2, tol=0.001, C=10):
    y2 = Y[i2]
    alpha2 = alphas[i2]
    if alphas[i2] > 0 and alphas[i2] < C:
        E2 = E[i2]
    else:
        E2 = f(X, Y, alphas, X[i2], b[0])-y2
    r2 = E2*y2
    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        i1 = secondheuristic(alphas, E, E2, i2, C)
        if i1 != None:
            if takestep(Y, X, alphas, i1, i2, b, E):
                return 1
        for i1 in range(len(alphas)):
            if i1 != i2 and alphas[i1] > 0 and alphas[i1] < C:
                if takestep(Y, X, alphas, i1, i2, b, E):
                    return 1
        for i1 in range(len(alphas)):
            if i1 != i2 and takestep(Y, X, alphas, i1, i2, b, E):
                return 1
    return 0

def run(X, Y, C=10):
    N = len(Y)
    alphas = np.zeros(N)
    E = np.zeros(N)
    b = [0.]
    for i in range(N):
        E[i] = f(X, Y, alphas, X[i], b[0]) - Y[i]
    numChanged = 0
    examineAll = 1
    while numChanged > 0 or examineAll == 1:
        numChanged = 0
        if examineAll == 1:
            for i in range(N):
                numChanged += examineExample(X, Y, alphas, b, E, i)
        else:
            for i in range(N):
                if alphas[i] > 0 and alphas[i] < C:
                    numChanged += examineExample(X, Y, alphas, b, E, i)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
    return (alphas, b[0])

def predict(X, Y, alphas, b, x):
    res = f(X, Y, alphas, x, b)
    if res > 0:
        return 1
    else:
        return 0
