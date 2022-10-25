
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
#delete
data.drop(columns=["User ID", "Gender"], axis=1, inplace=True)

data_frame = data.values
data_frame[:, 1] = data_frame[:, 1]/1000
N, d = data_frame.shape
x = data_frame[:, 0:d-1].reshape(-1, d-1)
y = data_frame[:, d-1].reshape(-1, 1)
true_x = []
true_y = []
false_x = []
false_y = []

for item in data_frame:
    if item[2] == 1:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])
'''
plt.scatter(true_x, true_y, marker='o', c = 'b')
plt.scatter(false_x, false_y, marker='s', c ='r')
plt.show()
'''

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def decisionBoundary(p):
    if p >= 0.5:
        return 1
    else:
        return 0

def predict(features, weights): #ánh xạ theo hàm sigmoid (0, 1)
    z = np.dot(features, weights) #(400, 1)
    return sigmoid(z) 

def cost_function(features, lables, weights): #cost - entropy : phi tuyến tính
    """
    features: (400, 3)
    lables: (400, 1)
    weights: (3, 1)
    return: cost_function
    """
    n = len(lables)
    predictions = predict(features, weights)
    """
    predictions: (400, 1)
    """
    cost_class1 = -lables*np.log(predictions) #y = 1 
    cost_class2 = -(1-lables)*np.log(1-predictions) # y = 0
    cost = cost_class1 + cost_class2
    return cost.sum()/n

def update_weight(features, lables, weights, learning_rate): #gradient decent
    """
    features: (400, 3)
    lables: (400, 1)
    weights: (3, 1)
    learning_rate: float
    return new_weight: float
    """
    n = len(lables)
    predictions = predict(features, weights) #(400,1)
    weights_temp = np.dot(features.T, (predictions-lables)) #(3,400)x(400,1)
    weights -= (weights_temp/n)*learning_rate
    return weights

def train(features, lables, weights, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weights = update_weight(features, lables, weights, learning_rate)
        cost = cost_function(features, lables, weights)
        cost_history.append(cost)
    return weights, cost_history

# Thêm cột 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x)) #features (400, 2) -> (400, 3)

weight = np.array([-11., 0.3, 0.035]).reshape(-1, 1) #(3, ) -> (3, 1)

# Số lần lặp bước 
iter = 100
#cost = np.zeros((iter, 1))
learning_rate = 0.00039
k = update_weight(x, y, weight, learning_rate)
print(k)
weight, loss = train(x, y, weight, learning_rate, iter)

yTime_series = np.array([i for i in range(iter)])
plt.plot(yTime_series,loss)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()

#Test
x_test = [1, 50, 76]
temp = predict(x_test, weight)
print(temp)

print("Value will be predicted for client who Age {} and EstimatedSalary {}000 ".format(x_test[1],x_test[2]))
if (decisionBoundary(temp)==1) :
    print("Predict value is {}. So this client will purchase!!".format(decisionBoundary(temp)))
else: 
    print("Predict value is {}. So this client will not purchase!!".format(decisionBoundary(temp)))
'''
plt.scatter(true_x,true_y,marker="o",c="b", edgecolors='none', s=30, label='purchase')
plt.scatter(false_x,false_y,marker="o",c="r", edgecolors='none', s=30, label='no purchase')
plt.legend(loc=1)
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.show()
'''

