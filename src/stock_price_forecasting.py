from pickletools import optimize

import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

df = pd.read_csv("../data/sp500.txt")
df = df.sort_values("Date")
data = df[["Close"]].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)

seq_length = 60

X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)
# model_lstm.save("model_lstm.h5")

model_gru = Sequential([
    GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
    BatchNormalization(),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])

model_gru.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
# model_gru.fit(X_train, y_train, epochs=10, batch_size=32)
# model_gru.save("model_gru.h5")

model_lstm = tf.keras.models.load_model("model_lstm.h5")
model_gru = tf.keras.models.load_model("model_gru.h5")

pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test))
pred_gru = scaler.inverse_transform(model_gru.predict(X_test))
y_test_inv = scaler.inverse_transform(y_test)

rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, pred_gru))

print("MRSE LSTM", rmse_lstm)
print("MRSE GRU", rmse_gru)

plt.plot(y_test_inv, label="Real")
plt.plot(pred_lstm, label="LSTM")
plt.plot(pred_gru, label="GRU")
plt.legend()
plt.title("Model comparission: LSTM vs GRU")
plt.savefig("../figures/compare-models-LSTM-GRU.png")
