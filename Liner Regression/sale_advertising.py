import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('advertising.csv')
#print(dataframe)
X = dataframe.values[:, 2] #toàn bộ hàng, lấy cột 2 (radio)
y = dataframe.values[:, 4]
#plt.scatter(X, y, marker='o')
#plt.show()

def predict(new_radio, weight, bias):
    return weight*new_radio + bias # y = mx + b

def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*X[i] + bias))**2 #tổng bình phương lỗi ESM
    return sum_error/n

def update_weight(X, y, weight, bias, learning_rate): #gradient descent
    n = len(X)
    weight_temp = 0
    bias_temp = 0
    for i in range (n):
        weight_temp += -2*X[i]*(y[i]-(X[i]*weight + bias)) #đạo hàm riêng theo weight
        bias_temp += -2*(y[i]-(X[i]*weight + bias)) #đạo hàm riêng theo bias
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate

    return weight, bias 

def train(X, y, weight, bias, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        cost = cost_function(X, y, weight, bias)
        cost_history.append(cost)

    return weight, bias, cost_history

weight, bias, cost = train(X, y, 0.03 , 0.0014, 0.001, 30)
print("ket qua: ")
print(weight)
print(bias)
print(cost)
print("gia tri du doan: ")
print((predict(19, weight, bias)))

solanlap = [i for i in range(30)]
plt.plot(solanlap, cost)
#plt.show()