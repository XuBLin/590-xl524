# --------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
# --------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import Seaborn_visualizer as SBV
import copy
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# ------------------------
# CODE PARAMETERS
# ------------------------

# USER PARAMETERS
IPLOT = True

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names, na_values='?', comment='\t',
                 sep=' ', skipinitialspace=True)


# ----------------------------------------
# VISUALIZE DATA
# ----------------------------------------


SBV.get_pd_info(df)
# SBV.pd_general_plots(df,HUE='Origin')
# SBV.pandas_2D_plots(df,col_to_plot=[1,4,5],HUE='Origin')

#----------------------------------------
#PRE-PROCESS DATA 
#(EXTRACT AND CONVERT TO TENSOR)
#----------------------------------------

print("----------------------")
print("EXTRACT DATA")
print("----------------------")

# SELECT COLUMNS TO USE AS VARIABLES
x_col = [1, 2, 3, 4, 5]
y_col = [0]
xy_col = x_col + y_col
x_keys = SBV.index_to_keys(df, x_col)        # dependent var
y_keys = SBV.index_to_keys(df, y_col)        # independent var
xy_keys = SBV.index_to_keys(df, xy_col)        # independent var

print("X=", x_keys)
print("Y=", y_keys)
# SBV.pd_general_plots(df[xy_keys])

# CONVERT SELECT DF TO NP
x = df[x_keys].to_numpy()
y = df[y_keys].to_numpy()

# REMOVE NAN IF PRESENT
xtmp = []
ytmp = []
for i in range(0, len(x)):
    if 'nan' not in str(x[i]) and 'nan' not in str(y[i]):
        xtmp.append(x[i])
        ytmp.append(y[i][0])
X = np.array(xtmp)
Y = np.array(ytmp)
# print(Y)

# NORMALIZE DATA
# print(np.mean(x_train,axis=0),np.std(x_train,axis=0))

X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
Y_max = np.max(Y, axis=0)
Y_min = np.min(Y, axis=0)

X_norm = (X - X_min) / (X_max - X_min)
Y_norm = (Y - Y_min) / (Y_max - Y_min)
# print(X_norm)
# print(Y_norm)

PARADIGM = 'batch'

model_type = "linear"
NFIT = 6

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
rand_indices = np.random.permutation(X_norm.shape[0])

CUT1 = int(f_train*X_norm.shape[0])
CUT2 = int((f_train+f_val)*X_norm.shape[0])
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:", train_idx.shape)
print("val_idx shape:", val_idx.shape)
print("test_idx shape:", test_idx.shape)

# ------------------------
# MODEL
# ------------------------


def model(x, p):
    global flag
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
    YPRED_T = model(X_norm[train_idx], p)
    YPRED_V = model(X_norm[val_idx], p)
    YPRED_TEST = model(X_norm[test_idx], p)
    MSE_T = np.mean((YPRED_T-Y_norm[train_idx])**2.0)
    MSE_V = np.mean((YPRED_V-Y_norm[val_idx])**2.0)

# ------------------------
# LOSS FUNCTION
# ------------------------


def loss(p, index_2_use):
    errors = model(X_norm[index_2_use], p)-Y_norm[index_2_use]  # VECTOR OF ERRORS
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
    tol = 10**-4            # EXIT AFTER CHANGE IN F IS LESS THAN THIS
    NDIM = len(paras)        # DIMENSION OF OPTIIZATION PROBLEM
    dparas = np.array([0.0]*NDIM)

    # OPTIMIZATION LOOP
    while iteration <= max_iter:
        prev = loss(paras, train_idx)
        loss_train.append(prev)
        loss_val.append(loss(paras, val_idx))
        for j in range(len(paras)):
            paras_copy = copy.deepcopy(paras)
            paras_copy[j] = paras_copy[j] - step_size
            loss_new = loss(paras_copy, train_idx)
            dparas[j] = (prev-loss_new) / step_size
        # print(dparas, prev, loss_new)
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
po = np.random.uniform(-2, -1., size=NFIT)
# TRAIN MODEL USING SCIPY MINIMIZ
p_final = minimizer(X_norm, Y_norm, po)
print("OPTIMAL PARAM:", p_final)


'''def loss2(p):
    errors = model(X_norm[train_idx], p)-Y_norm[train_idx]  # VECTOR OF ERRORS
    training_loss = np.mean(errors**2.0)                # MSE
    return training_loss


res = minimize(loss2, [0.5]*5, method='Nelder-Mead', tol=1e-5)
popt = res.x
print(popt)
print("actual miniloss:", loss(popt, train_idx))
'''

predict(p_final)


fig, ax = plt.subplots()
sequence = list(range(0, len(loss_train)))
ax.scatter(sequence, loss_train, color='blue', marker='.', label='Training loss')
ax.scatter(sequence, loss_val, color='orange', marker='.', label="Testing loss")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Optimizer iterations', fontsize=FS)
plt.ylabel('loss', fontsize=FS)

plt.show()

# ------------------------
# GENERATE PLOTS
# ------------------------

# PLOT TRAINING AND VALIDATION LOSS HISTORY
fig, ax = plt.subplots()
ax.scatter(X[train_idx, 0], Y[train_idx], color='blue', marker='.', label='Training')
ax.scatter(X[val_idx, 0], Y[val_idx], color='orange', marker='x', label="Validation")
ax.scatter(X[test_idx, 0], Y[test_idx], color='green', marker='*', label="Test")
ax.scatter(X[train_idx, 0], YPRED_T*(Y_max-Y_min)+Y_min, s=3, color='red', marker='.', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Cylinders', fontsize=FS)
plt.ylabel('MPG', fontsize=FS)

plt.show()


fig, ax = plt.subplots()
ax.scatter(X[train_idx, 1], Y[train_idx], color='blue', marker='.', label='Training')
ax.scatter(X[val_idx, 1], Y[val_idx], color='orange', marker='x', label="Validation")
ax.scatter(X[test_idx, 1], Y[test_idx], color='green', marker='*', label="Test")
ax.scatter(X[train_idx, 1], YPRED_T*(Y_max-Y_min)+Y_min, s=3, color='red', marker='.', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Displacement', fontsize=FS)
plt.ylabel('MPG', fontsize=FS)

plt.show()


fig, ax = plt.subplots()
ax.scatter(X[train_idx, 2], Y[train_idx], color='blue', marker='.', label='Training')
ax.scatter(X[val_idx, 2], Y[val_idx], color='orange', marker='x', label="Validation")
ax.scatter(X[test_idx, 2], Y[test_idx], color='green', marker='*', label="Test")
ax.scatter(X[train_idx, 2], YPRED_T*(Y_max-Y_min)+Y_min, s=3, color='red', marker='.', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Horsepower', fontsize=FS)
plt.ylabel('MPG', fontsize=FS)

plt.show()


fig, ax = plt.subplots()
ax.scatter(X[train_idx, 3], Y[train_idx], color='blue', marker='.', label='Training')
ax.scatter(X[val_idx, 3], Y[val_idx], color='orange', marker='x', label="Validation")
ax.scatter(X[test_idx, 3], Y[test_idx], color='green', marker='*', label="Test")
ax.scatter(X[train_idx, 3], YPRED_T*(Y_max-Y_min)+Y_min, s=3, color='red', marker='.', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Weight', fontsize=FS)
plt.ylabel('MPG', fontsize=FS)

plt.show()


fig, ax = plt.subplots()
ax.scatter(X[train_idx, 4], Y[train_idx], color='blue', marker='.', label='Training')
ax.scatter(X[val_idx, 4], Y[val_idx], color='orange', marker='x', label="Validation")
ax.scatter(X[test_idx, 4], Y[test_idx], color='green', marker='*', label="Test")
ax.scatter(X[train_idx, 4], YPRED_T*(Y_max-Y_min)+Y_min, s=3, color='red', marker='.', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Acceleration', fontsize=FS)
plt.ylabel('MPG', fontsize=FS)

plt.show()


fig, ax = plt.subplots()
ax.scatter(Y[train_idx], YPRED_T*(Y_max-Y_min)+Y_min, color='blue', marker='*', label='Training')
ax.scatter(Y[val_idx], YPRED_V*(Y_max-Y_min)+Y_min, color='orange', marker='*', label="Validation")
ax.scatter(Y[test_idx], YPRED_TEST*(Y_max-Y_min)+Y_min, color='green', marker='*', label="Test")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('y_data', fontsize=FS)
plt.ylabel('y_predict', fontsize=FS)

plt.show()

