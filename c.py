import numpy as np
import matplotlib.pyplot as plt

def closeForm(x, y):
    # return (x^T * x)^-1 * x^T * y

    x = np.column_stack((np.ones(len(x)), x))
    x_t = np.transpose(x)
    multiply = np.dot(x_t, x)
    m_inv = np.linalg.inv(multiply)
    return (np.dot(np.dot(m_inv, x_t), y))

def cost_func(theta, x, y):
    m = len(y)
    pd = np.dot(x, theta)
    return (1/2) * m * np.sum(np.square(pd - y))
    
def gradientDescent_batch(x, y, theta, it, lr):
    x = np.c_[np.ones((len(x),1)), x]
    y = np.array(y).reshape((len(y),1))
    m = len(y)
    cost_his = []
    for i in range(it):
        pd = np.dot(x, theta)
        theta = theta - (1 / m) * lr * ((x.T).dot((pd - y)))
        cost_his.append(cost_func(theta, x, y))
    return theta, cost_his

a, b, y = np.genfromtxt('dataset2.txt', unpack = True,  delimiter = ',' )
x = np.column_stack((a, b))
theta_c = closeForm(x, y)
print("CloseForm")
print("theta 0:\t{:0.5f}\ntheta 1:\t{:0.5f}\ntheta 2:\t{:0.5f}".format(theta_c[0], theta_c[1], theta_c[2]))
lr = 0.01
it = 1500
theta = [[0],[0],[0]]
theta_b, cost_his = gradientDescent_batch(x, y, theta, it, lr)
