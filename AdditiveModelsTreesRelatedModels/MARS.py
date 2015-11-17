'''
Created on 2014-7-15

@author: xiajie
'''
import numpy as np
import ozone_data
from LinearRegression import leastsquare
import copy

'''
Just like splines(full name:Multivariate Adaptive RegressionSplines),
 but is more local and very efficient for computation.
Suitable for high dimension data.
'''

class H:
    def __init__(self, h=None, directions=[], t_idxes=[], xt=True):
        self.h = h
        self.directions = directions  #p
        self.t_idxes = t_idxes        #i
        self.xt = xt

def model2matix(Model):
    M = []
    for model in Model:
        h = model.h
        M.append(h)
    return M

def modelknotsnum(Model):
    knots = []
    for model in Model:
        for p in range(len(model.directions)):
            for i in range(len(model.t_idxes)):
                knots.append((model.directions(p),model.t_idxes(i)))
    s = set(knots)
    return len(s)

def basis_pair(x, t):
    if x > t:
        x_t = x-t
        t_x = 0
    else:
        x_t = 0
        t_x = t-x
    return x_t, t_x

def create_basis_matrix(X):
    N = len(X)
    P = len(X[0])
    CXT = np.zeros((P,N,N))
    CTX = np.zeros((P,N,N))
    for p in range(P):
        #t
        for i in range(N):
            #x
            for j in range(N):
                x_t, t_x = basis_pair(X[j,p], X[i,p])
                CXT[p,i,j] = x_t
                CTX[p,i,j] = t_x
    return CXT, CTX

def forward(CXT, CTX, Y):
    N = len(Y)
    P = len(CXT)
    h0 = H(np.ones(N).tolist())
    Model = [h0]
    while True:
        min_error = 999999999.
        best_h_1 = None
        best_h_2 = None
        best_i = None
        best_p = None
        best_model = None
        for m in Model:
            model = np.array(m.h)
            for p in range(P):
                for i in range(N):
                    X = model2matix(Model)
                    x_t = CXT[p,i]
                    t_x = CTX[p,i]
                    new_h_1 = x_t*model
                    new_h_2 = t_x*model
                    X.append(new_h_1.tolist())
                    X.append(new_h_2.tolist())
                    X = np.transpose(np.array(X))
                    #print X
                    beta = leastsquare.ls(X, Y)
                    if beta == None:
                        continue
                    error = leastsquare.RSS(X, Y, beta)
                    if error < min_error:
                        print "error:", error
                        min_error = error
                        best_h_1 = new_h_1
                        best_h_2 = new_h_2
                        best_i = i
                        best_p = p
                        best_model = m
        if best_model == None:
            continue
        directions = copy.deepcopy(best_model.directions)
        indexes = copy.deepcopy(best_model.t_idxes)
        directions.append(best_p)
        indexes.append(best_i)
        print "d:",directions
        print "ti:", indexes
        Model.append(H(best_h_1.tolist(), directions, indexes, True))
        Model.append(H(best_h_2.tolist(), directions, indexes, False))
        if len(Model) >= 100 or min_error < 100.:
            break
    return Model

def predict(beta, Model, x, X_train):#X_train are knots here
    h = []
    for model in Model:
        if len(model.directions) == 0:
            h.append(1)
            continue
        product = 1.
        for p,xp in enumerate(x):
            if p not in model.directions:
                continue
            i = model.t_idxes[model.directions.index(p)]
            t = X_train[i][p]
            #print "predict:", i
            x_t, t_x = basis_pair(xp, t)
            if model.xt == True:
                product *= x_t
            else:
                product *= t_x
        h.append(product)
    return np.transpose(beta).dot(np.array(h))

def GCV(Model, Y, lmbda, beta):
    N = len(Y)
    X = model2matix(Model)
    X = np.transpose(np.array(X))
    Y_hat = leastsquare.predict(X, beta)
    rss = np.sum((Y-Y_hat)**2)
    denominator = (1-float(lmbda)/N)**2
    return rss/denominator

def removed_rss(Model, index, Y):
    X = model2matix(Model)
    del X[index]
    X = np.transpose(np.array(X))
    if len(X) <= 1:
        return 999999999., None
    beta = leastsquare.ls(X, Y)
    Y_hat = leastsquare.predict(X, beta)
    rss = np.sum((Y-Y_hat)**2)
    return rss, beta

def backward(Model, Y):
    Models = []
    betas = []
    rsses = []
    #Models.append(copy.deepcopy(Model))
    while True:
        if len(Model) < 2:
            break
        min_error = 999999999.
        best_i = None
        best_beta = None
        best_rss = None
        for i in range(len(Model)):
            error, beta = removed_rss(Model, i, Y)
            if error < min_error:
                min_error = error
                best_i = i
                best_rss = error
                best_beta = beta
        del Model[best_i]
        Models.append(copy.deepcopy(Model))
        betas.append(best_beta)
        rsses.append(best_rss)
        #print best_rss, best_beta
    min_gcv = 999999999.
    best_i = None
    for i in range(len(Models)):
        lmbda = len(Models[i])+3*modelknotsnum(Models[i])
        gcv = GCV(Models[i], Y, lmbda, betas[i])
        if gcv < min_gcv:
            min_gcv = gcv
            best_i = i
    return Models[best_i]

def fortest(X_train, Model):
    for index in range(len(X_train)):
        x = X_train[index]
        h = []
        for model in Model:
            if len(model.directions) == 0:
                h.append(1)
                continue
            product = 1.
            for p,xp in enumerate(x):
                if p not in model.directions:
                    continue
                i = model.t_idxes[model.directions.index(p)]
                t = X_train[i][p]
                #print "predict:", i
                x_t, t_x = basis_pair(xp, t)
                if model.xt == True:
                    product *= x_t
                else:
                    product *= t_x
                h.append(product)

if __name__ == '__main__':
    data = ozone_data.load()
    data_train, data_test = ozone_data.traintest(data)
    X_train, Y_train = ozone_data.cook(data_train)
    X_test, Y_test = ozone_data.cook(data_test)
    CXT, CTX = create_basis_matrix(X_train)
    Model = forward(CXT, CTX, np.array(Y_train))
    #Model, beta = backward(Model, Y_train)
    X = model2matix(Model)
    X = np.transpose(np.array(X))
    beta = leastsquare.ls(X,Y_train)
    print "last error:", leastsquare.RSS(X, Y_train, beta)
    print beta
    fortest(X_train, Model)
    #err = 0.
    #for i in range(len(Y_test)):
    #    res = predict(beta, Model, X_test[i], X_train)
    #    print res, Y_test[i]
    #    err += (res-Y_test[i])**2
    #print "test error:", err
    
    #print "**********************************"
    #ls_beta = leastsquare.ls(X_train, Y_train)
    #print ls_beta
    #res = leastsquare.predict(X_test, ls_beta)
    #for l in range(len(Y_test)):
    #    print res[l], Y_test[l]
