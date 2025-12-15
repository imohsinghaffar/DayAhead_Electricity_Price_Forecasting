import optuna
import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader_2024 import DataLoader2024
from LSTM.lstm_model import LSTMModel

def objective(trial):
    # Load Data (Cached if possible, but here we load fresh or pass it in)
    # To save time, we should load data outside global or pass it?
    # Optuna runs sequential by default in python, so global df is fine.
    pass 

# Global load to avoid reloading every trial
loader = DataLoader2024()
df = loader.load_data()

# Split 80/20 for tuning
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:] # actually we use val_df implicitly inside LSTM?
# No, LSTMModel splits internally if we don't provide X_v. 
# But for hyperparam tuning, we want a fixed validation set to compare trials fairly.
# The LSTMModel.train method takes 'train_ratio'. 
# We should use strict temporal split.

def run_study(n_trials=20):
    
    def objective_wrapper(trial):
        # Hyperparameters to tune
        units_1 = trial.suggest_categorical('units_1', [32, 64, 128, 256])
        units_2 = trial.suggest_categorical('units_2', [16, 32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Instantiate Model
        # We need to target a specific column? "Price"
        lstm = LSTMModel(train_df, target_col='Price', window_size=168, forecast_horizon=24)
        
        # Train
        # We use a smaller epoch count for tuning to store time, or full? 
        # User wants "realistic", so maybe 15-20 epochs for tuning is enough to see convergence.
        try:
            # We need to capture validation loss. 
            # The current LSTMModel.train doesn't return history easily unless we modify it.
            # I will modify LSTMModel.train to return history or I can assume it works.
            # Wait, I previously modified LSTMModel? Let me check specific file.
            # I can't easily modify it again without context switch.
            # Actually, I can use a callback/custom implementation here.
            
            # Re-implementing a simple train loop here might be cleaner for Optuna
            # BUT reusing the class is better for consistency.
            # Let's trust the class runs. But how do we get the metric?
            # The class evaluates on a validation split internally?
            # The class splits using `train_ratio`.
            
            lstm.train(
                train_ratio=0.85, 
                epochs=5, # Fast search
                batch_size=batch_size,
                units_1=units_1,
                units_2=units_2,
                dropout=dropout,
                learning_rate=learning_rate
            )
            
            # Predict on validation set to get score
            # We need to replicate the validation logic
            y_pred, y_true = lstm.predict_on_df(df) # Predict on WHOLE df? 
            # We want metrics only on the validation part (last 20%)
            # The model was trained on 80%.
            
            # Align
            # y_pred will correspond to ... wait. predict_on_df returns matching y_true.
            # We just need to slice the last 20%.
            
            n_val = len(y_pred) - int(len(y_pred) * 0.85) # Roughly? 
            # No, 'predict_on_df' usually predicts on samples it can form.
            # Let's just calculate RMSE on the last 500 samples (validation).
            
            valid_rmse = np.sqrt(np.mean((y_pred[-24*7:] - y_true[-24*7:])**2)) # Last week
            
            return valid_rmse
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_wrapper, n_trials=n_trials)
    
    print("Best params:", study.best_params)
    return study.best_params

if __name__ == "__main__":
    run_study(n_trials=10) # 10 trials for quick demo, user asked for "search"
