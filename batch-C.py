import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['ggplot'])

data = np.loadtxt('dataset3.txt', delimiter=',', usecols=range(3))

X=data[:,0:2]
y=data[:,2]
print(X)
#y=np.array(y).reshape((47,1))


# x1 = np.linspace(6, 8, 5, endpoint=True)
# y1 = np.linspace(20, 25, 5, endpoint=True)
# x1=np.array(x1).reshape((5,1))
# y1=np.array(y1).reshape((5,1))
# X=np.concatenate((X,x1),axis=0)
# y=np.concatenate((y,y1),axis=0)
#
#

X_b = np.c_[np.ones((len(X),1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


def cal_cost(theta, X, y):

    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1500):

    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 3))
    for it in range(iterations):
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * ((X.T).dot((prediction - y)))
        #theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)
        print(theta)
    return theta,cost_history#, theta_history

#theta = np.random.randn(j, 1)

lr =0.01
n_iter = 1500

theta = [0,0,0]
theta=np.array(theta).reshape((3,1))


X_b = np.c_[np.ones((len(X),1)),X]
theta,cost_history= gradient_descent(X_b,y,theta,lr,n_iter)
#,cost_history,theta_history

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f},\nTheta2:          {:0.3f}'.format(theta[0][0],theta[1][0],theta[2][0]))
#print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
plt.plot(range(n_iter),cost_history)
plt.show()
# X_new = np.array([[4],[23]])
# X_new_b = np.c_[np.ones((2,1)),X_new]
# y_predict = X_new_b.dot(theta)
#
#
# X=data[:,0:1]
# y=data[:,1:2]
# x1 = np.linspace(6, 8, 5, endpoint=True)
# y1 = np.linspace(20, 25, 5, endpoint=True)
# x1=np.array(x1).reshape((5,1))
# y1=np.array(y1).reshape((5,1))
# X=np.concatenate((X,x1),axis=0)
# y=np.concatenate((y,y1),axis=0)
#
# x1 = np.linspace(20, 24, 5, endpoint=True)
# y1 = np.linspace(0, 10, 5, endpoint=True)
# x1=np.array(x1).reshape((5,1))
# y1=np.array(y1).reshape((5,1))
# X=np.concatenate((X,x1),axis=0)
# y=np.concatenate((y,y1),axis=0)
# X_b = np.c_[np.ones((len(X),1)),X]
# theta = [0,0]
# theta=np.array(theta).reshape((2,1))
# X_b = np.c_[np.ones((len(X),1)),X]
# theta2,cost_history2,theta_history2 = gradient_descent(X_b,y,theta,lr,n_iter)
# print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta2[0][0],theta2[1][0]))
# print('Final cost/MSE:  {:0.3f}'.format(cost_history2[-1]))
# myx=[6.2,12.8,22.1,30]
# for i in myx:
#     a1 = (theta2[1:] * i) + theta2[0]
#     print(a1)
#
# # fig,ax = plt.subplots(figsize=(12,8))
# # ax.set_ylabel('J(Theta)')
# # ax.set_xlabel('Iterations')
# # plt.plot(range(n_iter),cost_history)
#
#
# X_new2 = np.array([[4],[23]])
# X_new_b2 = np.c_[np.ones((2,1)),X_new2]
# y_predict2 = X_new_b2.dot(theta2)
#
#
# plt.plot(X_new,y_predict,'r-')
#
#
# plt.plot(X_new2,y_predict2,'r-')
# plt.plot(X,y,'b.')
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.show()