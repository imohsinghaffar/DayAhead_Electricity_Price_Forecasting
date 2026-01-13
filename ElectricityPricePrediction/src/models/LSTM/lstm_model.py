import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, df, target_col="Price", feature_cols=None, window_size=72):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols or [c for c in df.columns if c != target_col]
        self.input_cols = self.feature_cols + ([target_col] if target_col not in self.feature_cols else [])
        self.window_size = int(window_size)
        self.forecast_horizon = 24
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.model = None

    def build_model(self, input_shape, units_1=64, units_2=128, units_3=16, dropout=0.2):
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.LSTM(units_1, return_sequences=True)(inputs)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.LSTM(units_2, return_sequences=True)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.LSTM(units_3, return_sequences=False)(x)
        x = keras.layers.Dense(units_3, activation='relu')(x)
        outputs = keras.layers.Dense(self.forecast_horizon)(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss='huber', optimizer=keras.optimizers.Adam(0.001), metrics=['mae'])
        return model

    def train(self, epochs=50, batch_size=32):
        X_raw = self.df[self.input_cols].values
        y_raw = self.df[[self.target_col]].values
        X_scaled = self.x_scaler.fit_transform(X_raw)
        y_scaled = self.y_scaler.fit_transform(y_raw)
        
        Xs, ys = [], []
        for i in range(self.window_size, len(X_scaled) - self.forecast_horizon + 1):
            Xs.append(X_scaled[i-self.window_size:i])
            ys.append(y_scaled[i:i+self.forecast_horizon, 0])
        
        Xs, ys = np.array(Xs), np.array(ys)
        self.model = self.build_model((Xs.shape[1], Xs.shape[2]))
        self.model.fit(Xs, ys, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict_on_df(self, df_new):
        data_x = self.x_scaler.transform(df_new[self.input_cols].values)
        data_y = self.y_scaler.transform(df_new[[self.target_col]].values)
        Xs, _ = [], []
        for i in range(self.window_size, len(data_x)):
            Xs.append(data_x[i-self.window_size:i])
        Xs = np.array(Xs)
        if Xs.size == 0: return np.array([]), np.array([])
        preds = self.model.predict(Xs, verbose=0)
        return self.y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape), None
