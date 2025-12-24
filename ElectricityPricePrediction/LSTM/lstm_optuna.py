import os
import json
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow import keras

# Adjust paths to import from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_2024 import DataLoader2024
from LSTM.lstm_model import LSTMModel

optuna.logging.set_verbosity(optuna.logging.INFO)

def run_lstm_optimization(n_trials=20):
    print("Loading Data for Optimization...")
    loader = DataLoader2024()
    df = loader.load_data()
    
    # Split Data: Train (80%), Validation (20%)
    # Note: We use strict time split
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Train Size: {len(train_df)}")
    print(f"Val Size: {len(val_df)}")
    
    # Validation Context: We need previous window to predict start of val
    window_size = 168
    
    # Safety check for small sample data
    if len(train_df) < window_size + 24:
        print("WARNING: Dataset too small for 168h window. Reducing to 24h for testing.")
        window_size = 24
        
    val_context = pd.concat([train_df.iloc[-window_size:], val_df])
    
    def objective(trial):
        # Hyperparameters
        units_1 = trial.suggest_int('units_1', 32, 128, step=32)
        units_2 = trial.suggest_int('units_2', 16, 64, step=16)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Build Model
        model = LSTMModel(
            train_df, 
            window_size=window_size,
            forecast_horizon=24
        )
        
        # Train
        # We assume 10 epochs is enough for tuning to find promise, 
        # or use EarlyStopping inside (which is enabled by default in model.train)
        try:
            history = model.train(
                epochs=15, 
                batch_size=batch_size,
                units_1=units_1, 
                units_2=units_2, 
                dropout=dropout, 
                learning_rate=learning_rate,
                verbose=0
            ) 
        except Exception as e:
            # Prune failed runs (e.g. divergence)
            print(f"Pruning trial due to error: {e}")
            return float('inf')
            
        # Evaluate on Validation (Day-Ahead t+24)
        try:
            y_pred, y_true = model.predict_on_df(val_context)
            
            # y_pred is (N, 24). y_true is (N, 24).
            # We want RMSE of the 24th hour prediction (t+24)
            # index -1
            
            y_pred_24 = y_pred[:, -1]
            y_true_24 = y_true[:, -1]
            
            # Align lengths (predict_on_df might return slightly different length depending on window padding)
            # But here we padded correctly.
            
            rmse = np.sqrt(np.mean((y_true_24 - y_pred_24)**2))
            return rmse
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return float('inf')

    print(f"Starting Optuna Study ({n_trials} trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Study Completed.")
    print(f"Best RMSE: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    
    # Save Best Params
    output_path = os.path.join(os.path.dirname(__file__), 'best_lstm_params.json')
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"Saved best params to {output_path}")

if __name__ == "__main__":
    run_lstm_optimization()
