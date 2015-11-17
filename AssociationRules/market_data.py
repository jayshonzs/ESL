'''
Created on 2014-8-12

@author: xiajie
'''
import numpy as np

def kickNA(data):
    ret = []
    for i in range(len(data)):
        l = data[i]
        NA = False
        for j in range(len(l)):
            if l[j] == -1:
                NA = True
                break
        if NA == False:
            ret.append(l.tolist())
    return np.array(ret)

'''
1:income less, 2:income more
3:male, 4:female
5:married, 6:live 2gether, 7:divorce, 8:widowed, 9:single
10:young, 11:old
12:less education, 13:more education
14:professional, 15:sales, 16:worker, 17:service, 18:homemaker, 19:student, 20:military, 21:retired, 22:unemployed
23:live short, 24:live long
25:single, 26:dual income, 27:not dual income
28:less person in house, 29:more person in house
30:less young person in house, 31:more young person in house
32:own house, 33:rent house, 34:with parents
35:house, 36:condominium, 37:apartment, 38:mobile home, 39:other
40:us indian, 41:asian, 42:black, 43:east indian, 44:hispanic, 45:Pacific Islander, 46:White, 47:other
48:english, 49:spanish, 50:other
'''
def cookdata(data):
    N = len(data)
    P = len(data[0])
    Z = np.zeros((N, 50))
    for i in range(N):
        x = data[i]
        for j in range(P):
            if j == 0:
                if x[j] <= 5:
                    Z[i, 0] = 1
                else:
                    Z[i, 1] = 1
            elif j == 1:
                if x[j] == 1:
                    Z[i, 2] = 1
                else:
                    Z[i, 3] = 1
            elif j == 2:
                Z[i, 3+x[j]] = 1
            elif j == 3:
                if x[j] <= 4:
                    Z[i, 9] = 1
                else:
                    Z[i, 10] = 1
            elif j == 4:
                if x[j] <= 4:
                    Z[i, 11] = 1
                else:
                    Z[i, 12] = 1
            elif j == 5:
                Z[i, 12+x[j]] = 1
            elif j == 6:
                if x[j] <= 3:
                    Z[i, 22] = 1
                else:
                    Z[i, 23] = 1
            elif j == 7:
                Z[i, 23+x[j]] = 1
            elif j == 8:
                if x[j] <= 5:
                    Z[i, 27] = 1
                else:
                    Z[i, 28] = 1
            elif j == 9:
                if x[j] <= 5:
                    Z[i, 29] = 1
                else:
                    Z[i, 30] = 1
            elif j == 10:
                Z[i, 30+x[j]] = 1
            elif j == 11:
                Z[i, 33+x[j]] = 1
            elif j == 12:
                Z[i, 38+x[j]] = 1
            elif j == 13:
                Z[i, 46+x[j]] = 1
    return Z

def load(filename='marketing.data'):
    data = np.genfromtxt(filename, dtype=int, missing_values='NA', filling_values=-1)
    noNA = kickNA(data)
    return cookdata(noNA)

if __name__ == '__main__':
    data = load()
    print data
    print data.shape
