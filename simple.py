#DO NYOT CHANGE UWU DIS WORKEY 
import pandas as pd
from tqdm import tqdm
from rich.progress import track
import yfinance as yf
from datetime import date, timedelta

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras import layers

from sklearn.preprocessing import MinMaxScaler

ticker = 'AMZN' #'5253.T' #'TSLA'
interval = "1m"

start = date.today() - timedelta(days=7)
end = date.today()

data = yf.download(ticker, start=start, end=end,interval=interval)
df = pd.DataFrame(data)['Close']
df.to_csv(r'E:\ai\stock\stock\v2.1\simpledata.csv')

# Normalize the data
scaler = MinMaxScaler()
normalized_prices = scaler.fit_transform(df.values.reshape(-1,1))

# Prepare sequences and labels
SEQ_LEN = 60
sequences = []
labels = []

for i in range(len(normalized_prices) - SEQ_LEN):
    seq = normalized_prices[i:i+SEQ_LEN]
    label = normalized_prices[i+SEQ_LEN]
    sequences.append(seq)
    labels.append(label)

X = np.array(sequences)
y = np.array(labels)

def original_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(SEQ_LEN, 1), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model

model = original_model()    
model.fit(X, y, epochs=50, batch_size=32)

# Save the model
model.save(r"E:\ai\stock\stock\v2.1\128x5v64x3.h5")
