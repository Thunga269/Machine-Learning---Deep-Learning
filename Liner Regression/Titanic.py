
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv('train_titanic.csv')
data_test = pd.read_csv('test_titanic.csv')
#print(data_train.head(10))

#delete columns
data_train.drop(columns=["PassengerId", "Embarked", "Cabin", "Ticket", "Name"], axis=1, inplace=True)
data_test.drop(columns=["PassengerId", "Embarked", "Cabin", "Ticket", "Name"], axis=1, inplace=True)

#replace Male and Female
sex_mapping = {'male': 0, "female": 1}
data_train['Sex'] = data_train['Sex'].apply(lambda X: sex_mapping[X])
data_test['Sex'] = data_test['Sex'].apply(lambda X: sex_mapping[X])

#fill NAN
data_train["Age"].fillna(data_train.groupby('Sex')["Age"].transform("median"), inplace = True)
data_test["Age"].fillna(data_train.groupby('Sex')["Age"].transform("median"), inplace = True)
data_test["Fare"].fillna(method="pad", inplace=True)

X = data_train.values[:, 1:7]
y = data_train.values[:, 0] 
X_test = data_test.values[:, 0:6]

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
weight = [-0.08, 0.3, -0.002, 0.03, 0.05, -0.007]
bias = 0.802
weight, bias, cost = train(X, y, weight, bias, 0.0002, 100)

print("ket qua: ")
print(weight)
print(bias)
print("gia tri du doan: ")

#print((predict(X_test, weight, bias)))

solanlap = [i for i in range(100)]
plt.plot(solanlap, cost)
plt.show()
y_predict = predict(X_test, weight, bias)

#check accurancy

data_result = pd.read_csv('gender_submission.csv')
y_result = data_result[["Survived"]].values
predict = int(0)
for i in range (len(y_predict)):
  if y_predict[i] >= 0.5:
    if y_result[i] == 1:
      predict += 1
  if y_predict[i] < 0.5:
    if y_result[i] == 0:
      predict += 1
print("correct ratio:",predict / len(y_result))

