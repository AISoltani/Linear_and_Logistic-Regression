import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])


data = np.loadtxt('dataset1.txt', delimiter=',', usecols=range(2))

X=data[:,0:1]
y=data[:,1:2]


X_b = np.c_[np.ones((97,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


def cal_cost(theta, X, y):

    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


lr =0.01
n_iter = 1500
theta = [0,0]
theta=np.array(theta).reshape((2,1))
X_b = np.c_[np.ones((len(X),1)),X]

def stocashtic_gradient_descent(X, y, theta, learning_rate=0.01, iterations=1500):

    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind, :].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            prediction = np.dot(X_i, theta)

            theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost

    return theta, cost_history

theta,cost_history = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

X_new = np.array([[4],[23]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta)


plt.plot(range(n_iter),cost_history)

plt.figure()

plt.plot(X_new,y_predict,'r-')
plt.plot(X,y,'b.')
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.show()