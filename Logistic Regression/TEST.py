

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
#delete
data.drop(columns=["User ID", "Gender"], axis=1, inplace=True)

data_frame = data.values
N, d = data_frame.shape
x = data_frame[:, 0:d-1].reshape(-1, d-1)
y = data_frame[:, d-1].reshape(-1, 1)
x = np.hstack((np.ones((N, 1), dtype = int), x)) #features (400, 2) -> (400, 3)
#print(x)
#print(y)
weight = np.array([0., 0.03, -0.00003]).reshape(-1, 1) #(3, ) -> (3, 1)
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
print(predict(x, weight))