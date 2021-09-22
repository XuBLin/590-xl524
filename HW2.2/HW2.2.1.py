# --------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
# --------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import copy

# ------------------------
# CODE PARAMETERS
# ------------------------

# USER PARAMETERS
IPLOT = True

f = open('planar_x1_x2_x3_y.json')
t = json.load(f)
X1 = t['x1']
X2 = t['x2']
X3 = t['x3']
Y = np.array(t['y']).T
X = np.array([X1, X2, X3]).T


PARADIGM = 'batch'

model_type = "linear"
NFIT = 4

loss_train = []
loss_val = []


# ------------------------
# PARTITION DATA
# ------------------------
# TRAINING: DATA THE OPTIMIZER "SEES"
# VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
# TEST: NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)

f_train = 0.8
f_val = 0.15
f_test = 0.05

if(f_train+f_val+f_test != 1.0):
    raise ValueError("f_train+f_val+f_test MUST EQUAL 1")

# PARTITION DATA
rand_indices = np.random.permutation(X.shape[0])

CUT1 = int(f_train*X.shape[0])
CUT2 = int((f_train+f_val)*X.shape[0])
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:", train_idx.shape)
print("val_idx shape:", val_idx.shape)
print("test_idx shape:", test_idx.shape)

# ------------------------
# MODEL
# ------------------------


def model(x, p):
    if model_type == "linear":
        pred = []
        for i in range(x.shape[0]):
            sum = 0
            for j in range(x.shape[1]):
                sum = sum + p[j]*x[i][j]
            sum = sum + p[-1]
            pred.append(sum)
        pred = np.array(pred).T
        return pred
    if model_type == "logistic":
        pred = []
        for i in range(x.shape[0]):
            sum = 0
            for j in range(x.shape[1]):
                sum = sum + p[j]*x[i][j]
            sum = sum + p[-1]
            pred.append(1.0/(1.0+np.exp(-sum)))
        pred = np.array(pred).T
        return pred

# FUNCTION TO MAKE VARIOUS PREDICTIONS FOR GIVEN PARAMETERIZATION


def predict(p):
    global YPRED_T, YPRED_V, YPRED_TEST, MSE_T, MSE_V
    YPRED_T = model(X[train_idx], p)
    YPRED_V = model(X[val_idx], p)
    YPRED_TEST = model(X[test_idx], p)
    MSE_T = np.mean((YPRED_T-Y[train_idx])**2.0)
    MSE_V = np.mean((YPRED_V-Y[val_idx])**2.0)

# ------------------------
# LOSS FUNCTION
# ------------------------


def loss(p, index_2_use):
    errors = model(X[index_2_use], p)-Y[index_2_use]  # VECTOR OF ERRORS
    training_loss = np.mean(errors**2.0)                # MSE
    return training_loss

# ------------------------
# MINIMIZER FUNCTION
# ------------------------


def minimizer(X, Y, initial_guess, algo='GD', LR=0.01):
    global epoch, epochs, loss_train, loss_val
    # x0=initial guess, (required to set NDIM)
    # algo=GD or MOM
    # LR=learning rate for gradient decent

    # PARAM
    paras = initial_guess
    iteration = 1            # ITERATION COUNTER
    step_size = 0.001            # STEP SIZE FOR FINITE DIFFERENCE
    max_iter = 1000        # MAX NUMBER OF ITERATION
    tol = 10**-7            # EXIT AFTER CHANGE IN F IS LESS THAN THIS
    NDIM = len(paras)        # DIMENSION OF OPTIIZATION PROBLEM
    dparas = np.array([0]*NDIM)

    # OPTIMIZATION LOOP
    while iteration <= max_iter:
        prev = loss(paras, train_idx)
        loss_train.append(prev)
        loss_val.append(loss(paras, val_idx))
        for j in range(len(paras)):
            paras_copy = copy.deepcopy(paras)
            paras_copy[j] = paras_copy[j] - step_size
            dparas[j] = (prev-loss(paras_copy, train_idx))/step_size
            # print(j)
        paras = paras - dparas * LR

        iteration = iteration + 1
        # print(iteration)
        delta = abs(prev - loss(paras, val_idx))
        if delta < tol:
            print('Stop Iteration')
            return paras
    return paras


# ------------------------
# FIT MODEL
# ------------------------

# RANDOM INITIAL GUESS FOR FITTING PARAMETERS
# from 1 to 2
po = np.random.uniform(2, 1., size=NFIT)
# po = np.array([2.718, 3.14, 1])
# TRAIN MODEL USING SCIPY MINIMIZ
p_final = minimizer(X, Y, po)
print("OPTIMAL PARAM:", p_final)
# print("length", len(loss_train))
predict(p_final)


fig, ax = plt.subplots()
sequence = list(range(0, len(loss_train)))
ax.scatter(sequence, loss_train, color='red', marker='.', label='Training loss')
ax.scatter(sequence, loss_val, color='green', marker='.', label="Testing loss")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Optimizer iterations', fontsize=FS)
plt.ylabel('loss', fontsize=FS)

plt.show()

# #------------------------
# #DOUBLE CHECK PART-1 OF HW2.1
# #------------------------

# x=np.array([[3],[1],[4]])
# y=np.array([[2,5,1]])

# A=np.array([[4,5,2],[3,1,5],[6,4,3]])
# B=np.array([[3,5],[5,2],[1,4]])
# print(x.shape,y.shape,A.shape,B.shape)
# print(np.matmul(x.T,x))
# print(np.matmul(y,x))
# print(np.matmul(x,y))
# print(np.matmul(A,x))
# print(np.matmul(A,B))
# print(B.reshape(6,1))
# print(B.reshape(1,6))
