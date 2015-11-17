'''
Created on 2014-7-15

@author: xiajie
'''
import numpy as np
import ozone_data

class Box:
    def __init__(self, mins=None, maxs=None, sets=None, mean=None):
        self.mins = mins
        self.maxs = maxs
        self.sets = sets
        self.mean = mean
    def isinbox(self, x):
        for i in range(len(x)):
            if self.mins[i] > x[i]:
                return False
            elif self.maxs[i] < x[i]:
                return False
        return True
    def isinboxexcept(self, x, direction):
        for i in range(len(x)):
            if i == direction:
                continue
            if self.mins[i] > x[i]:
                return False
            elif self.maxs[i] < x[i]:
                return False
        return True

def peeling(box, direction, length, left=True):
    X_peeled = []
    Y_peeled = []
    sets = box.sets
    for i in range(len(sets[0])):
        if left == True:
            if sets[0][i][direction] > box.mins[direction]+length:
                Y_peeled.append(sets[1][i])
                X_peeled.append(sets[0][i])
        elif left == False:
            if sets[0][i][direction] < box.maxs[direction]-length:
                Y_peeled.append(sets[1][i])
                X_peeled.append(sets[0][i])
    return np.array(X_peeled), np.array(Y_peeled)

def topdown(X_train, Y_train, result_boxes, alpha=0.05):
    mins = [m for m in [X_train[:,i].min() for i in range(len(X_train[0]))]]
    maxs = [m for m in [X_train[:,i].max() for i in range(len(X_train[0]))]]
    box = Box(mins, maxs, (X_train, Y_train), np.mean(Y_train))
    result_boxes.append(box)
    while True:
        max_increase = 0.
        best_peeled_X = None
        best_peeled_Y = None
        for direction in range(len(X_train[0])):
            length = (maxs[direction] - mins[direction])*alpha
            #left
            left_X_peeled, left_Y_peeled = peeling(box, direction, length, left=True)
            left_peeled_mean = np.mean(left_Y_peeled)
            #right
            right_X_peeled, right_Y_peeled = peeling(box, direction, length, left=False)
            right_peeled_mean = np.mean(right_Y_peeled)
            
            if left_peeled_mean > right_peeled_mean:
                peeled_mean = left_peeled_mean
                increase = peeled_mean - box.mean
                if increase > max_increase:
                    max_increase = increase
                    best_peeled_X = left_X_peeled
                    best_peeled_Y = left_Y_peeled
            else:
                peeled_mean = right_peeled_mean
                increase = peeled_mean - box.mean
                if increase > max_increase:
                    max_increase = increase
                    best_peeled_X = right_X_peeled
                    best_peeled_Y = right_Y_peeled
        if best_peeled_X == None or best_peeled_Y == None:
            break
        mins = [m for m in [best_peeled_X[:,i].min() for i in range(len(best_peeled_X[0]))]]
        maxs = [m for m in [best_peeled_X[:,i].max() for i in range(len(best_peeled_X[0]))]]
        new_box = Box(mins, maxs, (best_peeled_X, best_peeled_Y), box.mean+max_increase)
        box = new_box
        result_boxes.append(box)
        if len(box.sets[0]) <= 3:
            break
    return box

def pasting(box, X_train, Y_train, direction, length, left=True):
    X_pasted = []
    Y_pasted = []
    for i in range(len(X_train[0])):
        x = X_train[i]
        if box.isinboxexcept(x, direction) == False:
            continue
        if left == True:
            if x[direction] > box.mins[direction]-length:
                Y_pasted.append(Y_train[i])
                X_pasted.append(x)
        elif left == False:
            if x[direction] < box.maxs[direction]+length:
                Y_pasted.append(Y_train[i])
                X_pasted.append(x)
    for i in range(len(box.sets[0])):
        X_pasted.append(box.sets[0][i])
        Y_pasted.append(box.sets[1][i])
    return np.array(X_pasted), np.array(Y_pasted)

def downup(X_train, Y_train, min_box, result_boxes, alpha=0.05):
    box = result_boxes[-1]
    while True:
        max_increase = 0.1
        best_pasted_X = None
        best_pasted_Y = None
        for direction in range(len(X_train[0])):
            length = (box.maxs[direction] - box.mins[direction])*alpha
            #left
            left_X_pasted, left_Y_pasted = pasting(box, X_train, Y_train, direction, length, left=True)
            left_pasted_mean = np.mean(left_Y_pasted)
            #right
            right_X_pasted, right_Y_pasted = pasting(box, X_train, Y_train, direction, length, left=False)
            right_pasted_mean = np.mean(right_Y_pasted)
            
            if left_pasted_mean > right_pasted_mean:
                pasted_mean = left_pasted_mean
                increase = pasted_mean - box.mean
                if increase > max_increase:
                    max_increase = increase
                    best_pasted_X = left_X_pasted
                    best_pasted_Y = left_Y_pasted
            else:
                pasted_mean = right_pasted_mean
                increase = pasted_mean - box.mean
                if increase > max_increase:
                    max_increase = increase
                    best_pasted_X = right_X_pasted
                    best_pasted_Y = right_Y_pasted
        if best_pasted_X == None or best_pasted_Y == None:
            break
        if max_increase <= 0.2:
            break
        mins = [m for m in [best_pasted_X[:,i].min() for i in range(len(best_pasted_X[0]))]]
        maxs = [m for m in [best_pasted_X[:,i].max() for i in range(len(best_pasted_X[0]))]]
        new_box = Box(mins, maxs, (best_pasted_X, best_pasted_Y), box.mean+max_increase)
        box = new_box
        result_boxes.append(box)
    return

def cutout(X_train, Y_train, box):
    X_left = []
    Y_left = []
    for i in range(len(X_train)):
        x = X_train[i].tolist()
        if box.isinbox(x) == False:
            X_left.append(x)
            Y_left.append(Y_train[i])
    return np.array(X_left), np.array(Y_left)

def build_model(X_train, Y_train, X_test, Y_test):
    model = []
    while True:
        result_boxes = []
        min_box = topdown(X_train, Y_train, result_boxes)
        downup(X_train, Y_train, min_box, result_boxes)
        #cv
        min_error = 999999999.
        best_box = None
        for box in result_boxes:
            quad_error = 0.
            for i,x in enumerate(X_test):
                if box.isinbox(x):
                    res = box.mean
                    quad_error += (res-Y_test[i])**2
            if quad_error < min_error:
                min_error = quad_error
                best_box = box
        if box != None:
            X_train, Y_train = cutout(X_train, Y_train, best_box)
            model.append(best_box)
        if len(X_train) <= 3 and len(X_train) > 0:
            mins = [m for m in [X_train[:,i].min() for i in range(len(X_train[0]))]]
            maxs = [m for m in [X_train[:,i].max() for i in range(len(X_train[0]))]]
            last_box = Box(mins, maxs, (X_train, Y_train), np.mean(Y_train))
            model.append(last_box)
            break
        if len(X_train) == 0:
            break
    return model

def predict(model, x):
    for box in model:
        if box.isinbox(x) == True:
            return box.mean
    return None

if __name__ == '__main__':
    data = ozone_data.load()
    data_train, data_test = ozone_data.traintest(data)
    X_train, Y_train = ozone_data.cook(data_train)
    X_test, Y_test = ozone_data.cook(data_test)
    model_boxes = build_model(X_train, Y_train, X_test, Y_test)
    for box in model_boxes:
        for i in range(len(box.mins)):
            print "(%f:%f)" % (box.mins[i],box.maxs[i]),
        print " mean:%f" % box.mean
    error = 0.
    for i in range(len(X_train)):
        res = predict(model_boxes, X_train[i])
        print res, Y_train[i]
        error += (res-Y_train[i])**2
    print error/len(Y_train)
