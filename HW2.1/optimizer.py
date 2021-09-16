import json
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import random


class Data:
    def __init__(self, json_path):
        f = open(json_path)
        t = json.load(f)
        self.age = t['x']
        self.weight = t['y']
        self.isadult = t['is_adult']


wt_data = Data('/home/xubolin/590-CODES/DATA/weight.json')
# copy = Data('weight.json')


def logis_model(x, A, w, x0, S):  # x is array
    return [S + A/(1+np.exp((x0-x[i])/w)) for i in range(len(x))]


def logis_mse(x, y, A, w, x0, S):   # x and y are arrays
    pred = logis_model(x, A, w, x0, S)
    return mean_squared_error(y, pred)


# Normalizing data for logisitic regression
min_age = np.min(wt_data.age)
max_age = np.max(wt_data.age)
norm_age = (wt_data.age - min_age) / (max_age - min_age)

min_weight = np.min(wt_data.weight)
max_weight = np.max(wt_data.weight)
norm_weight = (wt_data.weight - min_weight) / (max_weight - min_weight)

# train test split // label no need
age_train, age_test, weight_train, weight_test, label_train, label_test =\
            train_test_split(norm_age, norm_weight, wt_data.isadult,
                             test_size=0.2, random_state=128)


# initial
A, w, x0, S = 0.5, 0.5, 0.5, 0.5

# tol = 10**-9
t, tmax = 0, 1000
step_size = 0.001
LR = 0.1
train_loss = []
test_loss = []
x_axis = []


def GDAlgo(A, w, x0, S, dA, dw, dx0, dS, LR):
    Anew = A - LR * dA
    wnew = w - LR * dw
    x0new = x0 - LR * dx0
    Snew = S - LR * dS
    return Anew, wnew, x0new, Snew


def GDplusAlgo(A, w, x0, S, dA, dw, dx0, dS, LR, gradient=[], gamma=0):
    if len(gradient) != 0:
        dA = gamma * gradient[0] + LR * dA
        dw = gamma * gradient[1] + LR * dw
        dx0 = gamma * gradient[2] + LR * dx0
        dS = gamma * gradient[3] + LR * dS
        gradient = np.array([dA, dw, dx0, dS])
        Anew = A - dA
        wnew = w - dw
        x0new = x0 - dx0
        Snew = S - dS
    else:
        Anew = A - LR * dA
        wnew = w - LR * dw
        x0new = x0 - LR * dx0
        Snew = S - LR * dS
    return Anew, wnew, x0new, Snew


def RMSPropAlgo(A, w, x0, S, dA, dw, dx0, dS, LR, decay_rate=0, eps=0):
    global cache
    sum = dA**2 + dw**2 + dx0**2 + dS**2
    cache = decay_rate * cache + (1-decay_rate) * sum
    RMSLR = LR/np.sqrt(cache+eps)
    Anew = A - RMSLR * dA
    wnew = w - RMSLR * dw
    x0new = x0 - RMSLR * dx0
    Snew = S - RMSLR * dS
    return Anew, wnew, x0new, Snew


def AlgoSelect(Algo, A, w, x0, S, dA, dw, dx0, dS, LR, gradient=[], gamma=0,
               decay_rate=0, eps=0):
    global cache
    if Algo == 'GD':
        return GDAlgo(A, w, x0, S, dA, dw, dx0, dS, LR)
    elif Algo == 'GD+':
        return GDplusAlgo(A, w, x0, S, dA, dw, dx0, dS, LR, gradient, gamma)
    elif Algo == 'RMSProp':
        return RMSPropAlgo(A, w, x0, S, dA, dw, dx0, dS, LR, decay_rate, eps)


def Caldevs(prev, x, y, A, w, x0, S, step_size):
    dA = (prev - logis_mse(x, y, A-step_size, w, x0, S))/step_size
    dw = (prev - logis_mse(x, y, A, w-step_size, x0, S))/step_size
    dx0 = (prev - logis_mse(x, y, A, w, x0-step_size, S))/step_size
    dS = (prev - logis_mse(x, y, A, w, x0, S-step_size))/step_size
    return dA, dw, dx0, dS


def optimizer(xtrain, ytrain, xtest, ytest, LR=0.1):
    method = input("Please input the method(batch/mini-batch/stochastic)^^")
    algo = input("Please input the algorithm(GD/GD+/RMSProp)")
    global A, w, x0, S, train_loss, test_loss, x_axis, cache
    gamma, decay_rate, cache, eps = 0, 0, 0, 0
    prevdelta = -1
    if algo == 'GD+':
        gamma = input("Please input the value of gamma^^")
    elif algo == 'RMSProp':
        decay_rate = input("Please input the value of decay_rate^^")
        decay_rate = float(decay_rate)
        cache = 0
        eps = 10**-8
    # initial
    if method == 'stochastic':
        sequence = list(range(0, len(age_train)))
        index = random.sample(sequence, k=len(sequence))
    A, w, x0, S = 0.5, 0.5, 0.5, 0.5

    tol = 10**-7
    t, tmax = 0, 500
    step_size = 0.001
    LR = 0.01
    train_loss = []
    test_loss = []
    x_axis = []
    while t < tmax:
        gradient = []
        t = t + 1
        test_loss.append(logis_mse(xtest, ytest, A, w, x0, S))
        x_axis.append(t)
        if method == 'batch':
            prev = logis_mse(xtrain, ytrain, A, w, x0, S)
            train_loss.append(prev)
            dA, dw, dx0, dS = Caldevs(prev, xtrain, ytrain, A, w, x0, S, step_size)
            Anew, wnew, x0new, Snew = AlgoSelect(algo, A, w, x0, S, dA, dw, dx0,
                                                 dS, LR, gradient, gamma, decay_rate, eps)
            A, w, x0, S = Anew, wnew, x0new, Snew
        elif method == 'mini-batch':
            sequence = list(range(0, len(age_train)))
            index = random.sample(sequence, k=len(age_train)//2)
            xbatch = np.array([xtrain[i] for i in index])
            ybatch = np.array([ytrain[i] for i in index])
            prev = logis_mse(xbatch, ybatch, A, w, x0, S)
            dA, dw, dx0, dS = Caldevs(prev, xbatch, ybatch, A, w, x0, S, step_size)
            Anew, wnew, x0new, Snew = AlgoSelect(algo, A, w, x0, S, dA, dw, dx0,
                                                 dS, LR, gradient, gamma, decay_rate, eps)
            A, w, x0, S = Anew, wnew, x0new, Snew
            # choose another batch
            xbatch = np.array([xtrain[i] for i in sequence if i not in index])
            xbatch = np.array([ytrain[i] for i in sequence if i not in index])
            prev = logis_mse(xbatch, ybatch, A, w, x0, S)
            dA, dw, dx0, dS = Caldevs(prev, xbatch, ybatch, A, w, x0, S, step_size)
            Anew, wnew, x0new, Snew = AlgoSelect(algo, A, w, x0, S, dA, dw, dx0,
                                                 dS, LR, gradient, gamma, decay_rate, eps)
            train_loss.append(prev)
            A, w, x0, S = Anew, wnew, x0new, Snew
        elif method == 'stochastic':
            '''sequence = list(range(0, len(age_train)))
            index = random.sample(sequence, k=1)
            xbatch = np.array([xtrain[i] for i in index])
            ybatch = np.array([ytrain[i] for i in index])
            prev = logis_mse(xbatch, ybatch, A, w, x0, S)
            train_loss.append(prev)'''
            
            for j in range(len(sequence)):
                x = np.array([xtrain[index[j]]])
                y = np.array([ytrain[index[j]]])
                prev = logis_mse(x, y, A, w, x0, S)
                dA, dw, dx0, dS = Caldevs(prev, x, y, A, w, x0, S, step_size)
                Anew, wnew, x0new, Snew = AlgoSelect(algo, A, w, x0, S, dA, dw, dx0,
                                                     dS, LR, gradient, gamma, decay_rate, eps)
                A, w, x0, S = Anew, wnew, x0new, Snew
            train_loss.append(prev)
        else:
            raise ValueError("method must be batch, mini-batch or stochastic")

        delta = abs(prev - logis_mse(xtrain, ytrain, Anew, wnew, x0new, Snew))

        if abs(delta-prevdelta) < tol:
            print('Stop iteration')
            break

        prevdelta = delta


optimizer(age_train, weight_train, age_test, weight_test, LR=0.1)

fig, ax = plt.subplots()
ax.scatter(x_axis, train_loss, color='red', marker='.', label='Training loss')
ax.scatter(x_axis, test_loss, color='green', marker='.', label="Testing loss")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Optimizer iterations', fontsize=FS)
plt.ylabel('loss', fontsize=FS)

plt.show()

# plot ydata and ypredicted
fig, ax = plt.subplots()

train_pred = logis_model(age_train, A, w, x0, S)
test_pred = logis_model(age_test, A, w, x0, S)

ax.scatter(train_pred, weight_train, color='yellow', marker='.', label='Training set')
ax.scatter(test_pred, weight_test, color='blue', marker='.', label="Test set")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Predicted', fontsize=FS)
plt.ylabel('Actual', fontsize=FS)

plt.show()

# plot denormalized age and weight
fig, ax = plt.subplots()

denorm_age_train = age_train * (max_age - min_age) + min_age
denorm_age_test = age_test * (max_age - min_age) + min_age
denorm_weight_train = weight_train * (max_weight - min_weight) + min_weight
denorm_weight_test = weight_test * (max_weight - min_weight) + min_weight
norm_pred_all = np.array(logis_model(norm_age, A, w, x0, S))
denorm_pred_all = norm_pred_all * (max_weight - min_weight) + min_weight

ax.scatter(denorm_age_train, denorm_weight_train, color='green', marker='.', label='Training set')
ax.scatter(denorm_age_test, denorm_weight_test, color='blue', marker='.', label="Testing set")
ax.plot(wt_data.age, denorm_pred_all, 'r-', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Age', fontsize=FS)
plt.ylabel('Weight', fontsize=FS)

plt.show()

