import json
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class Data:
    def __init__(self, json_path):
        f = open(json_path)
        t = json.load(f)
        self.age = t['x']
        self.weight = t['y']
        self.isadult = t['is_adult']


def Normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


weight = Data('/home/xubolin/590-CODES/DATA/weight.json')
copy = Data('/home/xubolin/590-CODES/DATA/weight.json')

copy_18 = Data('/home/xubolin/590-CODES/DATA/weight.json')

# Normalizing data for logisitic regression
min_copy_age = np.min(copy.age)
max_copy_age = np.max(copy.age)
norm_copy_age = (copy.age - min_copy_age) / (max_copy_age - min_copy_age)

min_copy_weight = np.min(copy.weight)
max_copy_weight = np.max(copy.weight)
norm_copy_weight = (copy.weight - min_copy_weight) / (max_copy_weight - min_copy_weight)

# train test split
age_train, age_test, weight_train, weight_test, label_train, label_test =\
            train_test_split(norm_copy_age, norm_copy_weight, copy.isadult,
            test_size=0.2, random_state=128)

############################################################
index = [i for i in range(len(copy_18.age)) if copy_18.age[i] < 18]
copy_18.age = [copy_18.age[i] for i in index]
copy_18.weight = [copy_18.weight[i] for i in index]

# Normalizing data for under 18 years old
min_18_age = np.min(copy_18.age)
max_18_age = np.max(copy_18.age)
norm_18_age = (copy_18.age - min_18_age) / (max_18_age - min_18_age)

min_18_weight = np.min(copy_18.weight)
max_18_weight = np.max(copy_18.weight)
norm_18_weight = (copy_18.weight - min_18_weight) / (max_18_weight - min_18_weight)

# train test split for under 18 year old
age_18_train, age_18_test, weight_18_train, weight_18_test =\
    train_test_split(norm_18_age, norm_18_weight, test_size=0.2)
age_18_train = np.array(age_18_train)
# print(type(age_18_train))


def linear_model(x, paras):
    return paras[0]*x + paras[1]
# paras[0] is m, paras[1] is b


def linear_mse(paras):
    weight_18_train_pred = linear_model(age_18_train, paras)
    return mean_squared_error(weight_18_train_pred, weight_18_train)


res = minimize(linear_mse, [0.5, 0.5], method='Nelder-Mead', tol=1e-5)
popt_linear = res.x
# print(popt_linear)


fig, ax = plt.subplots()
ax.plot(norm_18_age, norm_18_weight, 'o', label='Data')
# ax.plot(xe, ye, '-', label='Ground-Truth')
ax.plot(age_18_train, linear_model(age_18_train, popt_linear), 'k-', label="Model")

ax.legend()
FS = 10   # FONT SIZE
plt.xlabel('Normalized Age (years)', fontsize=FS)
plt.ylabel('Normalized Weight (lb) ', fontsize=FS)

plt.show()
######################################################################


def logis1_model(x, paras):
    return paras[3] + paras[0]/(1+np.exp((paras[2]-x)/paras[1]))


def logis1_mse(paras):
    weight_logis1_pred = logis1_model(age_train, paras)
    return mean_squared_error(weight_train, weight_logis1_pred)


res_logis1 = minimize(logis1_mse, [0.5, 0.5, 0.5, 0.5], method='Nelder-Mead', tol=1e-5)
popt_logis1 = res_logis1.x
# print(popt_logis1)

# the splited elements are not in order
age_train_sorted = sorted(age_train)

fig2, ax2 = plt.subplots()
ax2.plot(norm_copy_age, norm_copy_weight, 'o', label='Data')
ax2.plot(age_train_sorted, logis1_model(age_train_sorted, popt_logis1), 'k-', label="Model")
'''print(age_train)
print(logis1_model(age_train, popt_logis1))'''

ax2.legend()
FS = 10   # FONT SIZE
plt.xlabel('Normalized Age (years)', fontsize=FS)
plt.ylabel('Normalized Weight (lb) ', fontsize=FS)

plt.show()
#####################################################################


def logis2_mse(paras):
    isadult_logis1_pred = logis1_model(weight_train, paras)
    return mean_squared_error(label_train, isadult_logis1_pred)


res_logis2 = minimize(logis2_mse, [0.5, 0.5, 0.5, 0.5], method='Nelder-Mead', tol=1e-5)
popt_logis2 = res_logis2.x
# print(popt_logis2)

# the splited elements are not in order
# weight_train_sorted = sorted(weight_train)
xaxis = np.linspace(0, 1, 300)
# print(logis1_model(weight_train_sorted, popt_logis2))

fig3, ax3 = plt.subplots()
ax3.plot(norm_copy_weight, copy.isadult, 'o', label='Data')
# ax3.plot(weight_train_sorted, logis1_model(weight_train_sorted, popt_logis2), 'k-', label="Model")
ax3.plot(xaxis, logis1_model(xaxis, popt_logis2), 'k-', label="Model")


ax3.legend()
FS = 10   # FONT SIZE
plt.xlabel('Normalized Weight (lb)', fontsize=FS)
plt.ylabel('Adult=1 Child=0', fontsize=FS)

plt.show()

# print(popt_linear)
# print(popt_logis1)
#####################################################################

print('Predict weight by age and model 2')
pred_weight = logis1_model(norm_copy_age, popt_logis1)
denorm_weight = pred_weight * (max_copy_weight - min_copy_weight) + min_copy_weight
mse_weight = mean_squared_error(denorm_weight, copy.weight)
print('The first 10 element of predicted weight is', denorm_weight[0:10])
print('The MSE for the prediction is', mse_weight)

# print(popt_linear)
print('Predict age by weight and model 1')
pred_age = (norm_18_weight - popt_logis1[1]) / popt_logis1[0]
denorm_age = pred_age * (max_18_age - min_18_age) + min_18_age
mse_age = mean_squared_error(denorm_age, copy_18.age)
print('The first 10 element of predicted age is', denorm_age[0:10])
print('The MSE for the prediction is', mse_age)
# print(denorm_age)
# print(copy_18.age)

