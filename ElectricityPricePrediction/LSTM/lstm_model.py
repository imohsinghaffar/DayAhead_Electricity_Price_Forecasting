import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


class LSTMModel:
    def __init__(
        self,
        df,
        target_col="GBP/mWh",
        feature_cols=None,
        window_size=168,
        forecast_horizon=24,
        clip_outputs=False,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [
            c for c in df.columns if c != target_col
        ]

        # Include target in input for autoregression
        self.input_cols = (
            self.feature_cols
            + ([target_col] if target_col not in self.feature_cols else [])
        )

        self.window_size = int(window_size)
        self.forecast_horizon = int(forecast_horizon)
        self.clip_outputs = clip_outputs

        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _clean_array(self, arr, name="array"):
        arr = np.asarray(arr, dtype=np.float32)
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan + n_inf > 0:
            # last-resort cleanup
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _make_xy(self, X_scaled, y_scaled):
        X, y = [], []
        n = len(X_scaled)
        max_i = n - self.forecast_horizon + 1
        for i in range(self.window_size, max_i):
            X.append(X_scaled[i - self.window_size : i, :])
            y.append(y_scaled[i : i + self.forecast_horizon, 0])
        if not X:
            # No windows possible
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
              units_1=64, units_2=32, dropout=0.2, learning_rate=0.001):
        self.df = self.df.ffill().bfill().fillna(0)

        data_x = self._clean_array(self.df[self.input_cols].values, "X")
        data_y = self._clean_array(self.df[[self.target_col]].values, "y")

        split = int(len(self.df) * train_ratio)

        X_train_raw, X_val_raw = data_x[:split], data_x[split:]
        y_train_raw, y_val_raw = data_y[:split], data_y[split:]

        X_train = self.x_scaler.fit_transform(X_train_raw)
        y_train = self.y_scaler.fit_transform(y_train_raw)
        X_val = self.x_scaler.transform(X_val_raw)
        y_val = self.y_scaler.transform(y_val_raw)

        X_t, y_t = self._make_xy(X_train, y_train)
        X_v, y_v = self._make_xy(X_val, y_val)

        if X_t.shape[0] == 0:
            raise ValueError(
                "Not enough data to create training windows. "
                "Increase dataset size or reduce window_size/forecast_horizon."
            )

        self.model = self.build_model(
            (X_t.shape[1], X_t.shape[2]),
            units_1=units_1,
            units_2=units_2,
            dropout=dropout,
            learning_rate=learning_rate
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        if X_v.shape[0] == 0:
            # Train without validation set
            self.model.fit(
                X_t,
                y_t,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks,
                validation_split=0.1,
            )
        else:
            self.model.fit(
                X_t,
                y_t,
                validation_data=(X_v, y_v),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks,
            )

    def predict_on_df(self, df_new):
        """Predict on a df that includes the target; returns (y_pred, y_true)."""
        df_new = df_new.copy().ffill().bfill().fillna(0)

        data_x = self._clean_array(df_new[self.input_cols].values, "X_new")
        data_y = self._clean_array(df_new[[self.target_col]].values, "y_new")

        X_scaled = self.x_scaler.transform(data_x)
        y_scaled = self.y_scaler.transform(data_y)

        X, y_true_scaled = self._make_xy(X_scaled, y_scaled)
        if X.shape[0] == 0:
            raise ValueError(
                "Not enough data in df_new to create prediction windows. "
                "Need at least window_size + forecast_horizon rows."
            )

        pred_scaled = self.model.predict(X, verbose=0)
        pred_scaled = np.nan_to_num(pred_scaled)

        if self.clip_outputs:
            pred_scaled = np.clip(pred_scaled, 0.0, 1.0)

        y_pred = self.y_scaler.inverse_transform(pred_scaled)
        y_true = self.y_scaler.inverse_transform(y_true_scaled)

        return y_pred, y_true
