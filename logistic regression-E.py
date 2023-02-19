from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0 * z))))
    return G_of_Z


def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i] * theta[i]
    return Sigmoid(z)

CostHistory=[]
def Cost_Function(X, Y, theta, m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        hi = Hypothesis(theta, xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1 - Y[i]) * math.log(1 - hi)
        sumOfErrors += error
        CostHistory.append(sumOfErrors)
    const = -1 / m
    J = const * sumOfErrors

    #print('cost is ', J)
    return CostHistory


def Cost_Function_Derivative(X, Y, theta, j, m, alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta, X[i])
        error = (hi - Y[i]) * xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha) / float(m)
    J = constant * sumErrors
    return J


def Gradient_Descent(X, Y, theta, m, alpha):
    new_theta = []
    constant = alpha / m
    for j in range(len(theta)):
        CFDerivative = Cost_Function_Derivative(X, Y, theta, j, m, alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta


def newton(X, Y, alpha, theta, num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X, Y, theta, m, alpha)
        theta = new_theta
        if x % 100 == 0:
            Cost_Function(X, Y, theta, m)
            print('theta ', theta)
            print('cost is ', Cost_Function(X, Y, theta, m))
    Declare_Winner(theta)


def Declare_Winner(theta):
    score = 0
    winner = ""
    scikit_score = clf.score(X_test, Y_test)
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i], theta))
        answer = Y_test[i]
        if prediction == answer:
            score += 1
    my_score = float(score) / float(length)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

x_input1, x_input2, Y = np.genfromtxt('dataset3.txt', unpack=True, delimiter=',')
print(x_input1.shape)
print(x_input2.shape)
print(Y.shape)
X = np.column_stack((x_input1, x_input2))

X = min_max_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model= LogisticRegression()
model.fit(X_train, Y_train)
print('Acuraccy is: ', (clf.score(X_test, Y_test) * 100))
pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='g')
xlabel('score 1')
ylabel('score 2')
legend(['0', '1'])


initial_theta = [0, 0]
alpha = 0.01
iterations = 100
m = len(Y)
mycost=Cost_Function(X,Y,initial_theta,m)
mycost=np.asarray(mycost)
print(mycost.shape)
plt.figure()
plt.plot(range(iterations),mycost)
newton(X,Y,alpha,initial_theta,iterations)
# print("theta is: ",my_theta)
plt.show()
