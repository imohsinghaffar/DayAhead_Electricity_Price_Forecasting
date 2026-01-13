#!/usr/bin/env python3
"""
LSTM Optuna Hyperparameter Tuning Module
Uses Optuna to find optimal LSTM hyperparameters for electricity price forecasting.
"""
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Suppress TF warnings during tuning
tf.get_logger().setLevel('ERROR')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LSTMOptunaOptimizer:
    """Optuna-based hyperparameter optimizer for LSTM models."""
    
    def __init__(self, df, target_col='Price', feature_cols=None, split_idx=None):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols or [c for c in df.columns if c != target_col]
        self.split_idx = split_idx
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.best_params = None
        self.study = None
        self.tuning_history = []
        
        # Prepare data once
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare residuals for training."""
        if 'Price_lag_24' not in self.df.columns:
            self.df['Price_lag_24'] = self.df[self.target_col].shift(24)
        
        self.df['residual'] = self.df[self.target_col] - self.df['Price_lag_24']
        self.valid_df = self.df.dropna(subset=['residual'] + self.feature_cols)
        
        self.X_data = self.valid_df[self.feature_cols].values
        self.y_data = self.valid_df['residual'].values.reshape(-1, 1)
        
        self.X_scaled = self.scaler_X.fit_transform(self.X_data)
        self.y_scaled = self.scaler_y.fit_transform(self.y_data)
    
    def _build_model(self, input_shape, units_1, units_2, units_3, dropout, learning_rate):
        """Build LSTM model with given hyperparameters."""
        inputs = Input(shape=input_shape)
        
        x = LSTM(units_1, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        x = LSTM(units_2, return_sequences=True)(x)
        x = Dropout(dropout)(x)
        
        x = LSTM(units_3, return_sequences=False)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='huber')
        
        return model
    
    def _create_windows(self, X_scaled, y_scaled, window_size, train_end):
        """Create sliding windows for training."""
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        for i in range(window_size, len(self.valid_df)):
            if i < train_end:
                X_train.append(X_scaled[i-window_size:i])
                y_train.append(y_scaled[i])
            else:
                X_val.append(X_scaled[i-window_size:i])
                y_val.append(y_scaled[i])
        
        return (np.array(X_train), np.array(y_train), 
                np.array(X_val), np.array(y_val))
    
    def objective(self, trial):
        """Optuna objective function."""
        # Hyperparameters to tune
        window_size = trial.suggest_int('window_size', 48, 168, step=24)
        units_1 = trial.suggest_categorical('units_1', [32, 64, 128])
        units_2 = trial.suggest_categorical('units_2', [64, 128, 256])
        units_3 = trial.suggest_categorical('units_3', [8, 16, 32])
        dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Create windows
        train_end = int(len(self.valid_df) * 0.8)
        X_train, y_train, X_val, y_val = self._create_windows(
            self.X_scaled, self.y_scaled, window_size, train_end
        )
        
        if len(X_train) == 0 or len(X_val) == 0:
            return float('inf')
        
        # Build and train model
        model = self._build_model(
            input_shape=(window_size, len(self.feature_cols)),
            units_1=units_1,
            units_2=units_2,
            units_3=units_3,
            dropout=dropout,
            learning_rate=learning_rate
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        # Train with reduced epochs for speed
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )
        
        # Calculate validation RMSE
        y_pred_scaled = model.predict(X_val, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_val)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Log trial
        trial_info = {
            'trial_number': trial.number,
            'rmse': float(rmse),
            'params': {
                'window_size': window_size,
                'units_1': units_1,
                'units_2': units_2,
                'units_3': units_3,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            },
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss'])
        }
        self.tuning_history.append(trial_info)
        
        logger.info(f"LSTM Trial {trial.number}: RMSE={rmse:.4f}, units=[{units_1},{units_2},{units_3}], dropout={dropout:.2f}")
        
        # Clear memory
        tf.keras.backend.clear_session()
        
        return rmse
    
    def tune(self, n_trials=10):
        """Run hyperparameter tuning."""
        logger.info(f"Starting LSTM Optuna tuning with {n_trials} trials...")
        
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        # Compile results
        results = {
            'study_metadata': {
                'n_trials': n_trials,
                'best_rmse': self.study.best_value,
                'best_params': self.best_params,
                'best_trial_number': self.study.best_trial.number
            },
            'all_trials': self.tuning_history
        }
        
        logger.info(f"LSTM Optuna Complete. Best RMSE: {self.study.best_value:.4f}")
        logger.info(f"Best Params: {self.best_params}")
        
        return self.best_params, results
