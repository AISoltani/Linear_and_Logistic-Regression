import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1500):

    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * ((X.T).dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history

#theta = np.random.randn(j, 1)

lr =0.01
n_iter = 1500

theta = [0,0]
theta=np.array(theta).reshape((2,1))


X_b = np.c_[np.ones((len(X),1)),X]
theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)


print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))


def stocashtic_gradient_descent(X, y, theta, learning_rate=0.01, iterations=1500):

    m = len(y)
    cost_history = np.zeros(iterations)

    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            # rand_ind = np.random.randint(0, m)
            X_i = X[i, :].reshape(1, X.shape[1])
            y_i = y[i].reshape(1, 1)
            prediction = np.dot(X_i, theta)

            theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))
            cost += cal_cost(theta, X_i, y_i)
        cost_history[it] = cost

    return theta, cost_history



theta = [0,0]
theta=np.array(theta).reshape((2,1))


theta2,cost_history2 = stocashtic_gradient_descent(X_b,y,theta,lr,n_iter)

print('Theta0 online:          {:0.3f},\nTheta1 online:          {:0.3f}'.format(theta2[0][0],theta2[1][0]))
print('Final cost online/MSE:  {:0.3f}'.format(cost_history2[-1]))

#
# X_new = np.array([[4],[23]])
# X_new_b = np.c_[np.ones((2,1)),X_new]
# y_predict = X_new_b.dot(theta)
#
#

# plt.subplot(2,1,1)
# plt.plot(range(n_iter), cost_history, '-')
# plt.title('Compare Batch & Online')
# plt.ylabel('Cost Function Batch')
#
# plt.subplot(2,1,2)
# plt.plot(range(n_iter), cost_history2, '-')
# plt.xlabel('Iteration')
# plt.ylabel('Cost Function Online')
# plt.plot(range(n_iter), cost_history, linestyle='-', color='b')
# plt.plot(range(n_iter), cost_history2, linestyle='-', color='r')
# plt.xlabel('Iteration')
# plt.ylabel('Cost Function')
# plt.title('Compare Batch & Online')
# plt.show()
def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))


Xs, Ys = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-1, 4, 50))
Zs = np.array([compute_cost(X_b, y, theta) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = np.reshape(Zs, Xs.shape)

fig = plt.figure()#(figsize=(7, 7))
ax = fig.gca(projection="3d")
# ax.set_xlabel(r't0')
# ax.set_ylabel(r't1')
# ax.set_zlabel(r'cost')
# ax.view_init(elev=25, azim=40)
ax.plot_surface(Xs, Ys, Zs, cmap=cm.rainbow)

#
#
#plt.plot(range(n_iter),cost_history2)
#
# plt.figure()
# X_new = np.array([[4],[23]])
# X_new_b = np.c_[np.ones((2,1)),X_new]
# y_predict = X_new_b.dot(theta)
#
#
# plt.plot(X_new,y_predict,'r-')
# plt.plot(X,y,'b.')
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
plt.show()