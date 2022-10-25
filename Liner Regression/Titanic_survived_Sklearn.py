
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


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


model = LinearRegression() #chọn model
#thực hiện training
model.fit(X, y)
LinearRegression(copy_X=True, n_jobs=None, fit_intercept=True, normalize=False)

#return weight and bias
print("bias:", model.intercept_)
Coefficient = pd.DataFrame(model.coef_, columns=['Coefficient'])  
print("weight: ")
print(Coefficient) #6 features tương ứng 6 weights


#dự đoán
X_test = data_test.values[:, 0:6]
y_predic = model.predict(X_test) 
#print(y_predic)

#check accurancy
data_result = pd.read_csv('gender_submission.csv')
y_result = data_result[["Survived"]].values
predict = int(0)
for i in range (len(y_predic)):
  if y_predic[i] >= 0.5:
    if y_result[i] == 1:
      predict += 1
  if y_predic[i] < 0.5:
    if y_result[i] == 0:
      predict += 1
print("correct ratio:",predict / len(y_result))
