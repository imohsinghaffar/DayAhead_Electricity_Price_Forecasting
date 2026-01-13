import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

class ProbabilisticLSTM:
    def __init__(self, df, target_col='Price', feature_cols=None, window_size=72):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols or [c for c in df.columns if c != target_col]
        self.window_size = window_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def _build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = MCDropout(0.2)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = MCDropout(0.2)(x)
        x = LSTM(16, return_sequences=False)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='huber')
        return model

    def train(self, train_end_idx):
        """Train on residuals: actual - price(t-24)."""
        # Ensure Price_lag_24 exists
        if 'Price_lag_24' not in self.df.columns:
            self.df['Price_lag_24'] = self.df[self.target_col].shift(24)
        
        # Calculate residual
        self.df['residual'] = self.df[self.target_col] - self.df['Price_lag_24']
        valid_df = self.df.dropna(subset=['residual'] + self.feature_cols)
        
        # Adjust features: prioritize lags and cyclical
        X_data = valid_df[self.feature_cols].values
        y_data = valid_df['residual'].values.reshape(-1, 1)
        
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data)
        
        X_win, y_win = [], []
        # Relative to valid_df
        for i in range(self.window_size, len(valid_df)):
            X_win.append(X_scaled[i-self.window_size:i])
            y_win.append(y_scaled[i])
            
        X_win, y_win = np.array(X_win), np.array(y_win)
        self.model = self._build_model(X_win.shape[1:])
        
        # Training hyperparameters
        self.hyperparams = {
            'lstm_layers': [64, 128, 16],
            'dropout': 0.2,
            'batch_size': 32,
            'max_epochs': 100,
            'optimizer': 'adam',
            'loss_function': 'huber',
            'early_stopping_patience': 10,
            'lr_reduce_patience': 5,
            'validation_split': 0.1,
            'window_size': self.window_size
        }
        
        # More patient early stopping (increased to 20)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
        ]

        
        history = self.model.fit(
            X_win, y_win, 
            validation_split=0.1, 
            epochs=100, 
            batch_size=32, 
            verbose=1, 
            callbacks=callbacks
        )
        
        # Return training history for logging
        self.training_history = {
            'hyperparameters': self.hyperparams,
            'epochs_trained': len(history.history['loss']),
            'loss_history': history.history['loss'],
            'val_loss_history': history.history['val_loss'],
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'learning_rate_history': history.history.get('lr', [])
        }
        
        return self.training_history

    def predict(self, start_date, n_samples=50):
        """MCDropout inference: predicts residual then adds naive baseline."""
        if self.model is None: raise RuntimeError("Model not trained.")
        
        # Ensure lag and residual columns in full df for lookup
        if 'Price_lag_24' not in self.df.columns:
            self.df['Price_lag_24'] = self.df[self.target_col].shift(24)
            
        X_scaled = self.scaler_X.transform(self.df[self.feature_cols].values)
        test_mask = self.df.index >= start_date
        test_indices = self.df.index[test_mask]
        
        X_win = []
        naive_baselines = []
        for i in range(len(self.df)):
            if self.df.index[i] >= start_date:
                X_win.append(X_scaled[i-self.window_size:i])
                naive_baselines.append(self.df['Price_lag_24'].iloc[i])
        
        X_win = np.array(X_win)
        naive_baselines = np.array(naive_baselines).reshape(-1, 1)
        
        # MC Sampling (Residuals)
        mc_residuals = []
        for _ in range(n_samples):
            p_scaled = self.model.predict(X_win, verbose=0)
            p_real = self.scaler_y.inverse_transform(p_scaled)
            mc_residuals.append(p_real)
        
        mc_residuals = np.array(mc_residuals) # (samples, N, 1)
        
        # Add naive baseline back to each sample
        mc_preds = mc_residuals + naive_baselines[np.newaxis, :, :]
        
        res = pd.DataFrame(index=test_indices)
        res['prediction'] = mc_preds.mean(axis=0).flatten()
        res['uncertainty'] = mc_preds.std(axis=0).flatten()
        res['P05'] = np.percentile(mc_preds, 5, axis=0).flatten()
        res['P50'] = np.percentile(mc_preds, 50, axis=0).flatten()
        res['P95'] = np.percentile(mc_preds, 95, axis=0).flatten()
        
        return res

    def predict_next_24h(self, latest_df, n_samples=50):
        """Autoregressive forecast adding predicted residuals to naive baselines."""
        last_date = latest_df.index[-1]
        future_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=24, freq='h')
        
        current_data = latest_df.copy()
        future_preds = []
        
        for i in range(24):
            # Target hour: future_index[i]
            # Naive baseline for this hour: current_data[target_hour - 24h]
            target_hour = future_index[i]
            base_hour = target_hour - pd.Timedelta(hours=24)
            naive_base = current_data.loc[base_hour, self.target_col] if base_hour in current_data.index else current_data[self.target_col].iloc[-1]
            
            X_scaled = self.scaler_X.transform(current_data[self.feature_cols].tail(self.window_size).values)
            X_input = X_scaled.reshape(1, self.window_size, len(self.feature_cols))
            
            # MC Sample for Residual
            step_residuals = []
            for _ in range(n_samples):
                p_scaled = self.model.predict(X_input, verbose=0)
                p_real = self.scaler_y.inverse_transform(p_scaled)[0, 0]
                step_residuals.append(p_real)
            
            # Full predictions = residual + naive_base
            step_preds = np.array(step_residuals) + naive_base
            
            s_mean = np.mean(step_preds)
            future_preds.append({
                'prediction': s_mean, 
                'uncertainty': np.std(step_preds),
                'P05': np.percentile(step_preds, 5), 
                'P50': np.percentile(step_preds, 50), 
                'P95': np.percentile(step_preds, 95)
            })
            
            # Update current_data for next autoregressive step
            new_row = current_data.iloc[-1].copy()
            new_row.name = target_hour
            new_row[self.target_col] = s_mean
            current_data = pd.concat([current_data, pd.DataFrame([new_row])])
            
        return pd.DataFrame(future_preds, index=future_index)
