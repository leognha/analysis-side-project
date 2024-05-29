#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv("data/prices.csv")
df.head(5)

#%%

stock_prices = df['close'].values.reshape(-1, 1)
stock_prices

# Normalizing Data
scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices_normalized = scaler.fit_transform(stock_prices)


# Preparing Data for LSTM
X, y = [], []
for i in range(len(stock_prices_normalized) - 60):
    X.append(stock_prices_normalized[i:i + 60, 0])
    y.append(stock_prices_normalized[i + 60, 0])
X, y = np.array(X), np.array(y)
# %%
# 看