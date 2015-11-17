'''
Created on 2014-5-4

@author: xiajie
'''
import numpy as np
import correlation
import preprocess
import struct

def loaddata():
    inputs = np.genfromtxt('prostate.data', skip_header=1, dtype=float, usecols=(1,2,3,4,5,6,7,8))
    output = np.genfromtxt('prostate.data', skip_header=1, dtype=float, usecols=(9))
    Ttype = np.genfromtxt('prostate.data', skip_header=1, dtype='|S5', usecols=(10))
    #print inputs
    #print output
    #print Ttype
    return inputs, output, Ttype

def cookdata(data, output, Ttype):
    dlist = data.tolist()
    train_cooked = []
    test_cooked = []
    train_output = []
    test_output = []
    for i in range(len(dlist)):
        if Ttype[i] == 'T':
            train_cooked.append(dlist[i])
            train_output.append(output[i])
        else:
            test_cooked.append(dlist[i])
            test_output.append(output[i])
    train_data = standardize(np.array(train_cooked))
    test_data = standardize(np.array(test_cooked))
    train_output = np.array(train_output)
    test_output = np.array(test_output)
    return train_data,train_output,test_data,test_output

def cookdata2(data, output, Ttype):
    dlist = data.tolist()
    train_cooked = []
    test_cooked = []
    train_output = []
    test_output = []
    for i in range(len(dlist)):
        if Ttype[i] == 'T':
            train_cooked.append(dlist[i])
            train_output.append(output[i])
        else:
            test_cooked.append(dlist[i])
            test_output.append(output[i])
    train_data = np.array(train_cooked)
    test_data = np.array(test_cooked)
    train_output = np.array(train_output)
    test_output = np.array(test_output)
    return train_data,train_output,test_data,test_output

def calculate_correlation(data):
    cols = data.shape[1]
    cm = np.zeros((cols,cols))
    for i in range(cols):
        for j in range(cols):
            cm[i,j] = correlation.correlation(data[:, i], data[:, j])
    np.savetxt('correlation.txt', cm)
    return cm

def standardize(data):
    sdata = np.zeros(data.shape)
    for i in range(len(data[0])):
        column = data[:,i]
        EX = np.sum(column)/len(column)
        EX2 = np.sum(column*column)/len(column)
        V = np.sqrt(EX2-EX*EX)
        for j in range(len(column)):
            sdata[j, i] = (data[j, i]-EX)/V
    return sdata

def checkdata(train_data):
    for i in range(len(train_data[0])):
        column = train_data[:,i]
        EX = np.sum(column)/len(column)
        EX2 = np.sum(column*column)/len(column)
        print EX, EX2-EX*EX

def savebinary(X, Y):
    fx = open('binary_X.dat', 'wb')
    fy = open('binary_Y.dat', 'wb')
    for i in range(len(X)):
        for j in range(len(X[i])):
            fx.write(struct.pack("f",X[i, j]))
        fy.write(struct.pack("f", Y[i]))
    fx.close()
    fy.close()

def loadbinary(X, Y):
    #fx = open('binary_X.dat', 'rb')
    #for i in range(67):
    #    for j in range(8):
    #        print struct.unpack("f",fx.read(4))[0],
    #    print "."
    #print '############################################'
    #fxt = open('binary_XT.dat', 'rb')
    #for i in range(8):
    #    for j in range(67):
    #        print struct.unpack("f", fxt.read(4))[0],
    #    print "."
    #print '############################################'
    #fxadd = open('binary_X_SUB.dat', 'rb')
    #for i in range(67):
    #    for j in range(8):
    #        print struct.unpack("f",fxadd.read(4))[0],
    #    print "."
    #print '############################################'
    #M = np.transpose(X).dot(X)
    #print 'shape:', M.shape
    #for i in range(8):
    #    for j in range(8):
    #        print M[i, j],
    #    print "."
    #print '############################################'
    #fxtx = open('binary_XT_dot_X.dat', 'rb')
    #for i in range(8):
    #    for j in range(8):
    #        print struct.unpack("f",fxtx.read(4))[0],
    #    print "."
    #print '############################################'
    #XTY = np.transpose(X).dot(Y)
    #print 'shape:', XTY.shape
    #for i in range(8):
    #    print XTY[i],
    #print "."
    #print '############################################'
    #INV = np.linalg.inv(np.transpose(X).dot(X))
    #print 'shape:', INV.shape
    #for i in range(8):
    #    for j in range(8):
    #        print INV[i, j],
    #    print "."
    print '############################################'
    fm = open('MXT.dat', 'rb')
    for i in range(8):
        for j in range(67):
            print struct.unpack("f",fm.read(4))[0],
        print "."
    print '############################################'
    fbeta = open('beta.para', 'rb')
    for i in range(8):
        print struct.unpack("f",fbeta.read(4))[0],
    print "."

if __name__ != '__main__':
    inputs, output, Ttype = loaddata()
    train_data, train_out, test_data, test_out = cookdata(inputs, output, Ttype)
    calculate_correlation(train_data)
    sdata = standardize(train_data)
    checkdata(sdata)
else:
    inputs, output, Ttype = loaddata()
    train_data, train_out, test_data, test_out = cookdata2(inputs, output, Ttype)
    X, X_mean, Y_mean, X_std = preprocess.center_data(train_data, train_out, True)
    print '*********************'
    print len(X)
    #print X[0]
    #print X[1]
    print '*********************'
    #savebinary(X, train_out)
    loadbinary(X, train_out)
