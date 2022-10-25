import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
X = [[1, 2],
    [4, 5],
    [7, 8], 
    [10, 11]]
n = 4
y = [1, 2, 3, 4]
#weight = np.array([[0.03, 0.04]],dtype=float)
bias = 0.001
def predict(X, weight, bias):
    return np.dot(X, weight) + bias

def cost_funtion(X, y, weight, bias):
    sum_error = 0
    n = len(y)
    for i in range(n):
        sum_error += (y[i] - (np.dot(X[i], weight) + bias)) ** 2 #MSE: tổng bình phương lỗi
        #print(sum_error)
    return sum_error/n

def update_weight(X, y, weight, bias, learning_rate):
    X = np.asarray(X)
    n = len(y)
    k = len(X[1])
    bias_temp = 0
    weight_temp = np.zeros([k, ])
    for i in range(n):
        we = 0
        for j in range(k):
            we = -2*X[i][j]*(y[i]-(np.dot(X[i],weight)+ bias))
            weight_temp[j] += we
        bias_temp += -2*(y[i]-(np.dot(X[i],weight) + bias)) #đạo hàm riêng theo bias
    weight -= (weight_temp/n)* learning_rate

    bias -= (bias_temp/n)*learning_rate
    return weight, bias

def train(X, y, weight, bias, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        cost = cost_funtion(X, y, weight, bias)
        cost_history.append(cost)

    return weight, bias, cost_history

weight, bias, cost = train(X, y, [0.3, 0.4], 0.2, 0.001, 20)
solanlap = [i for i in range(20)]
plt.plot(solanlap, cost)
plt.show()