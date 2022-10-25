
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('client_purchased.csv')
#delete
data.drop(columns=["User ID", "Gender"], axis=1, inplace=True)
data = data.values
N, d = data.shape
data[:, 1] = data[:, 1]/1000
X = data[:, 0:d-1]
y = data[:, d-1]
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred=clf.predict([[35, 125]])
print(y_pred)
#return weight and bias
print("bias:", clf.intercept_)
print("weight: ", clf.coef_)

