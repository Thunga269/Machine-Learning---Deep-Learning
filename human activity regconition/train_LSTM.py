
import pandas as pd
import numpy as np

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

#doc du lieu

handswing_df = pd.read_csv("handswing.txt")
handapplaud_df = pd.read_csv("handapplaud.txt")
handwrite_df = pd.read_csv("handwrite.txt")
bodyswing_df = pd.read_csv("bodyswing.txt")

X=[]
y=[]
no_of_timestep = 10

dataset = handswing_df.iloc[:, 1:].values
n_sample = len(dataset)
#print(dataset[0])
for i in range(no_of_timestep, n_sample):
    X.append(dataset[i-no_of_timestep:i, :]) #mỗi lần input lấy 10 timesteps
    y.append([1, 0 , 0]) #hand swing

dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timestep, n_sample):
    X.append(dataset[i-no_of_timestep:i, :])
    y.append([0, 1, 0]) #body swing


dataset = handapplaud_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timestep, n_sample):
    X.append(dataset[i-no_of_timestep:i, :])
    y.append([0, 0, 1]) #hanwrite

X, y = np.array(X), np.array(y)
print(X.shape, y.shape) 
#(1773, 10, 132) (1773,) 1773=3*(600-10+1); 132: số lượng các tọa độ x, y , z, visible của các điểm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
#no_of_timesteps = X.shape[1]
#size_timestep = X.shape[2]
model.add(LSTM(units=50, return_sequences = True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#timestep: số bước trong 1 input, batchsize: đưa nhiều input vào một lần
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('model2.h5')

