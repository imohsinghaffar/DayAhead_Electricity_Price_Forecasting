"""
LSTM Optuna Optimization with Comprehensive Trial Logging

This script optimizes LSTM hyperparameters using Optuna and saves
detailed learning progress showing how the model improves over trials.

Features:
- Saves starting params and results for each trial
- Tracks improvement over time
- Saves comprehensive JSON with full learning history
- More epochs for better training
"""
import os
import json
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Adjust paths to import from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_2024 import DataLoader2024
from LSTM.lstm_model import LSTMModel

optuna.logging.set_verbosity(optuna.logging.INFO)


def run_lstm_optimization(n_trials=25, epochs_per_trial=30):
    """
    Run LSTM hyperparameter optimization with comprehensive logging.
    
    Args:
        n_trials: Number of Optuna trials (default 25)
        epochs_per_trial: Epochs for each trial training (default 30)
    """
    start_time = datetime.now()
    print("=" * 60)
    print("LSTM OPTUNA OPTIMIZATION")
    print("=" * 60)
    print(f"Start Time: {start_time.isoformat()}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per Trial: {epochs_per_trial}")
    print("=" * 60)
    
    # Load Data
    print("\nLoading Data for Optimization...")
    loader = DataLoader2024()
    df = loader.load_data()
    
    # Split Data: Train (80%), Validation (20%)
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Train Size: {len(train_df)}")
    print(f"Val Size: {len(val_df)}")
    print(f"Features: {df.shape[1] - 1}")
    
    # Validation Context
    window_size = 168
    if len(train_df) < window_size + 24:
        print("WARNING: Dataset too small for 168h window. Reducing to 24h.")
        window_size = 24
        
    val_context = pd.concat([train_df.iloc[-window_size:], val_df])
    
    # Track trial progress for logging
    trial_progress = []
    best_rmse_so_far = float('inf')
    
    def objective(trial):
        nonlocal best_rmse_so_far
        
        trial_start = datetime.now()
        
        # Hyperparameters Search Space
        units_1 = trial.suggest_int('units_1', 64, 256, step=64)
        units_2 = trial.suggest_int('units_2', 32, 128, step=32)
        units_3 = trial.suggest_int('units_3', 16, 64, step=16)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        window_size_trial = trial.suggest_categorical('window_size', [48, 72, 168])
        
        print(f"\n--- Trial {trial.number + 1}/{n_trials} ---")
        print(f"  units: {units_1}→{units_2}→{units_3}")
        print(f"  dropout: {dropout:.3f}, lr: {learning_rate:.6f}")
        print(f"  batch_size: {batch_size}, window: {window_size_trial}")
        
        # Build and Train Model
        model = LSTMModel(
            train_df, 
            window_size=window_size_trial,
            forecast_horizon=24
        )
        
        try:
            history = model.train(
                epochs=epochs_per_trial,
                batch_size=batch_size,
                units_1=units_1, 
                units_2=units_2,
                units_3=units_3,
                dropout=dropout, 
                learning_rate=learning_rate,
                verbose=0
            )
            
            # Get training metrics
            final_loss = history.history['loss'][-1] if 'loss' in history.history else None
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            
        except Exception as e:
            print(f"  ❌ Training failed: {e}")
            trial_progress.append({
                "trial_number": trial.number,
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return float('inf')
        
        # Evaluate on Validation
        try:
            y_pred, y_true = model.predict_on_df(val_context)
            y_pred_24 = y_pred[:, -1]
            y_true_24 = y_true[:, -1]
            rmse = np.sqrt(np.mean((y_true_24 - y_pred_24)**2))
            
        except Exception as e:
            print(f"  ❌ Evaluation failed: {e}")
            trial_progress.append({
                "trial_number": trial.number,
                "status": "EVAL_FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return float('inf')
        
        trial_end = datetime.now()
        duration = (trial_end - trial_start).total_seconds()
        
        # Check if this is the best so far
        is_best = rmse < best_rmse_so_far
        if is_best:
            improvement = ((best_rmse_so_far - rmse) / best_rmse_so_far * 100) if best_rmse_so_far != float('inf') else 100
            best_rmse_so_far = rmse
            print(f"  🎯 NEW BEST! RMSE: {rmse:.4f} (↓{improvement:.2f}%)")
        else:
            print(f"  RMSE: {rmse:.4f} (best so far: {best_rmse_so_far:.4f})")
        
        # Log trial details
        trial_progress.append({
            "trial_number": trial.number,
            "status": "COMPLETE",
            "rmse": float(rmse),
            "is_best_so_far": is_best,
            "best_rmse_at_this_point": float(best_rmse_so_far),
            "params": {
                "units_1": units_1,
                "units_2": units_2,
                "units_3": units_3,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "window_size": window_size_trial
            },
            "training_metrics": {
                "final_loss": float(final_loss) if final_loss else None,
                "final_val_loss": float(final_val_loss) if final_val_loss else None,
                "epochs_trained": len(history.history.get('loss', []))
            },
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        return rmse
    
    # Run Optimization
    print(f"\n{'='*60}")
    print(f"Starting Optuna Study ({n_trials} trials)...")
    print(f"{'='*60}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Print Final Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total Time: {total_duration/60:.1f} minutes")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save Best Params
    output_path = os.path.join(os.path.dirname(__file__), 'best_lstm_params.json')
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n✓ Saved best params to {output_path}")
    
    # Create Comprehensive Learning History
    learning_history = {
        "study_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_minutes": total_duration / 60,
            "n_trials": n_trials,
            "epochs_per_trial": epochs_per_trial,
            "data_shape": list(df.shape),
            "n_features": df.shape[1] - 1
        },
        "final_results": {
            "best_trial_number": study.best_trial.number,
            "best_rmse": float(study.best_value),
            "best_params": study.best_params,
            "trials_completed": len([t for t in trial_progress if t.get("status") == "COMPLETE"]),
            "trials_failed": len([t for t in trial_progress if "FAIL" in t.get("status", "")])
        },
        "learning_progress": {
            "first_trial": trial_progress[0] if trial_progress else None,
            "improvement_over_time": [],
            "last_trial": trial_progress[-1] if trial_progress else None
        },
        "all_trials": trial_progress
    }
    
    # Calculate improvement over time
    for i, trial in enumerate(trial_progress):
        if trial.get("status") == "COMPLETE":
            learning_history["learning_progress"]["improvement_over_time"].append({
                "trial": trial["trial_number"],
                "rmse": trial["rmse"],
                "best_so_far": trial["best_rmse_at_this_point"]
            })
    
    # Save comprehensive history
    history_path = os.path.join(os.path.dirname(__file__), 'optuna_learning_history.json')
    with open(history_path, 'w') as f:
        json.dump(learning_history, f, indent=4)
    print(f"✓ Saved learning history to {history_path}")
    
    # Also save to Analysis folder
    analysis_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'Analysis', 'optuna_lstm_history.json')
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, 'w') as f:
        json.dump(learning_history, f, indent=4)
    print(f"✓ Saved to Analysis folder: {analysis_path}")
    
    print("\n" + "=" * 60)
    print("Summary of Learning Progress:")
    print("=" * 60)
    if trial_progress:
        first_complete = next((t for t in trial_progress if t.get("status") == "COMPLETE"), None)
        if first_complete:
            print(f"First Trial RMSE: {first_complete['rmse']:.4f}")
            print(f"Final Best RMSE: {study.best_value:.4f}")
            improvement = ((first_complete['rmse'] - study.best_value) / first_complete['rmse'] * 100)
            print(f"Total Improvement: {improvement:.2f}%")
    
    return study.best_params


if __name__ == "__main__":
    # Run with more trials and epochs for better results
    run_lstm_optimization(n_trials=25, epochs_per_trial=30)
