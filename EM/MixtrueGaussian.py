'''
Created on 2014-8-27

@author: xiajie
'''
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

def simulate_data():
    data_set = []
    centers = [(0.18,0.4),(0.5,0.5),(0.82,0.6)]
    cov0 = [[0.01,.005],[.005,0.01]]
    cov1 = [[.01,-0.005],[-0.005,.01]]
    cov2 = [[0.01,.005],[.005,0.01]]
    cov = [cov0, cov1, cov2]
    num = [150, 200, 150]
    for k in range(len(centers)):
        center = centers[k]
        data = np.zeros((num[k], 2))
        print data.shape
        for i in range(num[k]):
            x = np.random.multivariate_normal(center, cov[k], 1)
            data[i] = x
        data_set.append(data)
    return data_set, 500

def norm(x, mu, cov):
    c = x - mu
    cov_i = np.linalg.inv(cov)
    cov_d = np.linalg.det(cov)
    D = len(x)
    normalization = 1.0/((2*np.pi)**D*cov_d)**0.5
    return normalization*np.exp(-0.5*(np.transpose(c).dot(cov_i).dot(c)))

def expectation(X, pi, mu, cov, K):
    responsibility = np.zeros((len(X), K))
    for i in range(len(X)):
        x = X[i]
        for k in range(K):
            numerator = pi[k]*norm(x, mu[k], cov[k])
            denominator = 0.
            for j in range(K):
                denominator += pi[j]*norm(x, mu[j], cov[j])
            res = numerator/denominator
            responsibility[i, k] = res
    return responsibility

def maximization(X, mu, resp, K=3):
    N = len(X)
    Nk = []
    D = len(X[0])
    new_cov = []
    new_pi = []
    for k in range(K):
        Nk.append(np.sum(resp[:, k], axis=0))
    for k in range(K):
        s = 0.
        for i in range(N):
            s += resp[i, k]*X[i]
        mu[k] = s/Nk[k]
    for k in range(K):
        s = np.zeros((D, D))
        for i in range(N):
            s += resp[i, k]*np.outer((X[i]-mu[k]), np.transpose((X[i]-mu[k])))
        new_cov.append(s/Nk[k])
    for k in range(K):
        new_pi.append(Nk[k]/N)
    
    return mu, new_cov, new_pi

def likelihood(X, mu, cov, pi, K=3):
    N = len(X)
    lkhood = 0.
    for i in range(N):
        x = X[i]
        inner = 0.
        for k in range(K):
            inner += pi[k]*norm(x, mu[k], cov[k])
        inner = np.log(inner)
        lkhood += inner
    return lkhood

def EM(X, K=3):
    cov = [np.array([[0.01, 0.], [0, 0.01]])]*3
    mu = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])
    pi = [0.33, 0.33, 0.34]
    lkhood = 0.
    while True:
        resp = expectation(X, pi, mu, cov, K)
        mu, cov, pi = maximization(X, mu, resp)
        new_lkhood = likelihood(X, mu, cov, pi)
        if np.abs(new_lkhood-lkhood) < 0.0001:
            break
        if new_lkhood > lkhood:
            lkhood = new_lkhood
    return mu, cov, pi, resp

#for testing simulated data
def cook_data(data_set):
    data = np.zeros((500,2))
    classes = np.zeros(len(data))
    idx = 0
    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            data[idx] = data_set[i][j]
            classes[idx] = i
            idx += 1
    return data, classes

def draw(X, classes, colored=True):
    mycm = mpl.cm.get_cmap('Paired')
    
    if colored == False:
        classes = np.zeros(len(classes))
    
    plt.scatter(X[:, 0], X[:, 1], s=50, c=classes, cmap=mycm)
    
    plt.show()
    
def max_c(resp):
    max_r = 0.
    best_i = None
    for i in range(len(resp)):
        if resp[i] > max_r:
            max_r = resp[i]
            best_i = i
    return best_i

if __name__ == '__main__':
    data, N = simulate_data();
    X, cls = cook_data(data);
    print data
    print cls
    draw(X, cls, False);
    mu, cov, pi, resp = EM(X)
    print "mu:"
    print mu
    print "cov0:"
    print cov[0]
    print "cov1:"
    print cov[1]
    print "cov2:"
    print cov[2]
    print pi
    
    for i in range(N):
        cls[i] = max_c(resp[i])
    draw(X, cls, True)
