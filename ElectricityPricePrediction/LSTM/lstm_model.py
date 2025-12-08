import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMModel:
    def __init__(self, df, target_col='GBP/mWh', feature_cols=None, window_size=24, forecast_horizon=24):
        """
        Initialize the LSTMModel.
        
        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
            target_col (str): The name of the target column.
            feature_cols (list): List of feature column names.
            window_size (int): The number of past time steps to use as input.
            forecast_horizon (int): The number of future time steps to predict.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [c for c in df.columns if c != target_col]
        # Ensure target is included in features for multivariate forecasting if needed, 
        # or handle separately. For now, we assume features include target lag if passed, 
        # but usually we construct X from all features including past target.
        if target_col not in self.feature_cols:
            self.feature_cols.append(target_col)
            
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        """
        Scales the data and creates sequences.
        """
        self.df.fillna(method='ffill', inplace=True)
        data = self.df[self.feature_cols].values
        self.data_scaled = self.scaler.fit_transform(data)
        return self.data_scaled

    def create_sequences(self, data):
        """
        Creates input sequences (X) and targets (y).
        """
        X, y = [], []
        # We want to predict the *next* forecast_horizon steps
        # So if we are at index i, input is [i-window_size : i], target is [i : i+forecast_horizon]
        # We need to ensure we have enough data
        
        target_idx = self.feature_cols.index(self.target_col)
        
        for i in range(self.window_size, len(data) - self.forecast_horizon + 1):
            X.append(data[i - self.window_size : i])
            y.append(data[i : i + self.forecast_horizon, target_idx])
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        Builds the LSTM/GRU model architecture.
        """
        model = keras.models.Sequential([
            # Convolutional layer to extract features from time series
            keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu", input_shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dense(self.forecast_horizon)
        ])
        
        model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        return model

    def train(self, train_split_ratio=0.8, epochs=20, batch_size=32):
        """
        Trains the model.
        """
        X, y = self.create_sequences(self.data_scaled)
        
        train_size = int(len(X) * train_split_ratio)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        print("LSTM training completed.")
        return history

    def predict(self, data_scaled=None):
        """
        Generates predictions.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet.")
            
        if data_scaled is None:
            data_scaled = self.data_scaled
            
        X, _ = self.create_sequences(data_scaled)
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform is tricky because scaler was fit on all features.
        # We need to construct a dummy array to inverse transform just the target.
        # Or we can create a separate scaler for target in __init__.
        # For simplicity, let's assume we use the min/max of the target column from the fitted scaler.
        
        target_idx = self.feature_cols.index(self.target_col)
        min_val = self.scaler.data_min_[target_idx]
        max_val = self.scaler.data_max_[target_idx]
        scale = max_val - min_val
        
        y_pred = y_pred_scaled * scale + min_val
        
        return y_pred

    def evaluate(self, test_start_idx):
        """
        Evaluates the model.
        """
        # This needs careful alignment with the sequences
        # For simplicity in this skeleton, we'll just return the predictions for now
        # and handle rigorous evaluation alignment in the comparison script.
        pass
