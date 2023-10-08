import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense

final_data = pd.read_csv("Final_Data_for_Model_training.csv")

features = final_data.iloc[:, 2:].values
target = final_data['Kp'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)  # Output layer with one unit for regression
])

model.compile(loss='mse', optimizer='adam')

model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

y_pred = model.predict(X_test_reshaped)

plt.figure(figsize=(12, 6))
plt.plot(final_data.index[-len(y_test):], y_test, label='Real Kp Values', color='blue')
plt.plot(final_data.index[-len(y_test):], y_pred, label='Predicted Kp Values', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Kp Value')
plt.legend()
plt.grid(True)
plt.title('Real vs. Predicted Kp Values (Test Period)')
plt.show()

model.save('DSCOVR')

model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

y_pred = model.predict(X_test_reshaped)

plt.figure(figsize=(12, 6))
plt.plot(final_data.index[-len(y_test):], y_test, label='Real Kp Values', color='blue')
plt.plot(final_data.index[-len(y_test):], y_pred, label='Predicted Kp Values', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Kp Value')
plt.legend()
plt.grid(True)
plt.title('Real vs. Predicted Kp Values (Test Period)')
plt.show()