import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(
        self,
        df,
        target_col="Price",
        feature_cols=None,
        window_size=168,
        forecast_horizon=24,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [
            c for c in df.columns if c != target_col
        ]

        # Input columns: Features + Target (autoregression)
        self.input_cols = (
            self.feature_cols
            + ([target_col] if target_col not in self.feature_cols else [])
        )

        self.window_size = int(window_size)
        self.forecast_horizon = int(forecast_horizon)

        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _clean_array(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _make_xy(self, X_scaled, y_scaled):
        X, y = [], []
        n = len(X_scaled)
        # We need a sequence of 'window_size' to predict 'forecast_horizon'
        # Range end: n - forecast_horizon + 1
        max_i = n - self.forecast_horizon + 1
        for i in range(self.window_size, max_i):
            X.append(X_scaled[i - self.window_size : i, :])
            # Target is the NEXT 'forecast_horizon' steps
            y.append(y_scaled[i : i + self.forecast_horizon, 0])
            
        if not X:
            return np.zeros((0, self.window_size, X_scaled.shape[1]), dtype=np.float32), \
                   np.zeros((0, self.forecast_horizon), dtype=np.float32)
                   
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def build_model(self, input_shape, units_1=64, units_2=32, dropout=0.2, learning_rate=0.001):
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.LSTM(units_1, return_sequences=True)(inputs)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.LSTM(units_2, return_sequences=False)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
        outputs = keras.layers.Dense(self.forecast_horizon)(x)

        model = keras.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
        model.compile(loss=keras.losses.Huber(), optimizer=opt, metrics=["mae"])
        return model

    def train(self, train_ratio=0.85, epochs=30, batch_size=32, 
              units_1=64, units_2=32, dropout=0.2, learning_rate=0.001, verbose=1):
        
        # 1. Strict Ffill
        self.df = self.df.ffill()

        data_x = self._clean_array(self.df[self.input_cols].values)
        data_y = self._clean_array(self.df[[self.target_col]].values)

        # Split Raw Data
        split = int(len(self.df) * train_ratio)
        X_train_raw, X_val_raw = data_x[:split], data_x[split:]
        y_train_raw, y_val_raw = data_y[:split], data_y[split:]

        # Fit Scalers on Train, Transform Both
        X_train = self.x_scaler.fit_transform(X_train_raw)
        y_train = self.y_scaler.fit_transform(y_train_raw)
        
        if len(X_val_raw) > 0:
            X_val = self.x_scaler.transform(X_val_raw)
            y_val = self.y_scaler.transform(y_val_raw)
        else:
            X_val, y_val = np.array([]), np.array([])

        # Make Windows
        X_t, y_t = self._make_xy(X_train, y_train)
        X_v, y_v = self._make_xy(X_val, y_val)

        if X_t.shape[0] == 0:
            raise ValueError("Dataset too small for windowing. Check data or reduce window_size.")

        # Build Model
        self.model = self.build_model(
            (X_t.shape[1], X_t.shape[2]),
            units_1=units_1,
            units_2=units_2,
            dropout=dropout,
            learning_rate=learning_rate
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]

        # Train
        if X_v.shape[0] > 0:
            history = self.model.fit(
                X_t, y_t,
                validation_data=(X_v, y_v),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                X_t, y_t,
                validation_split=0.1, # Fallback split if val set too small
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )

        return history

    def predict_on_df(self, df_new):
        """
        Takes a dataframe (with target column included for scaling/comparison purposes),
        returns (y_pred, y_true) where y_pred is (N, `forecast_horizon`)
        """
        df_new = df_new.copy().ffill()
        
        data_x = self._clean_array(df_new[self.input_cols].values)
        data_y = self._clean_array(df_new[[self.target_col]].values)

        X_scaled = self.x_scaler.transform(data_x)
        y_scaled = self.y_scaler.transform(data_y)

        X, y_true_scaled = self._make_xy(X_scaled, y_scaled)
        
        if X.shape[0] == 0:
            return np.array([]), np.array([])

        pred_scaled = self.model.predict(X, verbose=0)
        
        # Inverse Transform
        # Note: scaler expects (N, 1). We flatten prediction, inverse for each, or be clever.
        # Efficient way: Inverse transform assumes feature-wise. We only have 1 target feature.
        # But we have 'forecast_horizon' steps. 
        # y_scaler was fit on (N, 1).
        # pred_scaled is (N, horizon).
        # We can iterate or reshape.
        
        # Reshape to (N*Horizon, 1) -> Inverse -> Reshape back
        N, H = pred_scaled.shape
        y_pred = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(N, H)
        y_true = self.y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(N, H)

        return y_pred, y_true
