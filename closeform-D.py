import numpy as np
import matplotlib.pyplot as plt

def NormalizeData(data):
    max_data = np.max(data)
    min_data = np.min(data)
    for i in range(len(data)):
        data[i]= (data[i] - min_data) / (max_data - min_data)
    return data
x_input1,x_input2,y_output = np.genfromtxt('dataset2.txt', unpack = True,  delimiter = ',' )


x_input1=NormalizeData(x_input1)
x_input2=NormalizeData(x_input2)
y_output=NormalizeData(y_output)


x_input = np.column_stack((x_input1, x_input2))



temp_x = x_input
# j=np.append(x_input, z, axis=0)
# b = np.insert(x_input, 0, values=0, axis=1)
#x_input = np.concatenate((np.ones((97,1), dtype='float64'), x_input.reshape(97,1)), axis=1)
def closeform(x_input,y_output):
    ones = np.ones(len(x_input))
    x_input = np.column_stack((ones, x_input))
    xt_input = np.transpose(x_input)
    product = np.dot(xt_input, x_input)
    theInverse = np.linalg.inv(product)
    w = np.dot(np.dot(theInverse, xt_input), y_output)
    return w




# fiting
w=closeform(x_input,y_output)

print("Theta0:          ",w[0])
print("Theta1:          ",w[1])
print("Theta2:          ",w[2])

# X_new = np.array([[4],[23]])
# X_new_b = np.c_[np.ones((2,1)),X_new]
# y_predict = X_new_b.dot(w)
#

#Outlier, B Section
# x1 = np.linspace(6, 8, 5, endpoint=True)
# y1 = np.linspace(20, 25, 5, endpoint=True)
# x_input=np.concatenate((x_input,x1),axis=0)
# y_output=np.concatenate((y_output,y1),axis=0)
#
# x1 = np.linspace(20, 24, 5, endpoint=True)
# y1 = np.linspace(0, 10, 5, endpoint=True)
# x_input=np.concatenate((x_input,x1),axis=0)
# y_output=np.concatenate((y_output,y1),axis=0)
# print(x_input.shape)
# w2=closeform(x_input,y_output)
# predictions = []
# x_test = np.array(97)
# for i in (x_input):
#     components = w[1:] * i
#
#     predictions.append(sum(components) + w[0])
#
#
# print("Theta0:          ",w[0])
# print("Theta1:          ",w[1])
# print("Theta0:          ",w2[0])
# print("Theta1:          ",w2[1])
#
myx=[[1357,5],[2500,4]]

for i in myx:
    j=i
    a1 = (w[2] * j[0]) + (w[1]*j[1]) + (w[0])
    print(a1)

#
#
#
# X_new2 = np.array([[4],[23]])
# X_new_b2 = np.c_[np.ones((2,1)),X_new2]
# y_predict2 = X_new_b2.dot(w2)
#
#
# #plt.plot(range(len(y_output)),predictions)
#
# plt.figure()
# plt.plot(X_new,y_predict,'r-')
# plt.plot(X_new2,y_predict2,'r-')
# plt.plot(x_input,y_output,'b.')
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# #plt.axis([0,10,-2,8])
# plt.show()
