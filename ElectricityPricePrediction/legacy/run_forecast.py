import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI error se bachne ke liye
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os
import argparse
import traceback
import random
import warnings

# --- SETUP ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import DataLoader
from Naive.naive_baselines import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

def evaluate_robust(y_true, y_pred, model_name):
    """
    Robust evaluation: NaNs remove karta hai taake code crash na ho.
    """
    common_idx = y_true.index.intersection(y_pred.index)
    
    if len(common_idx) == 0:
        print(f"Warning: {model_name} ke liye koi common data nahi mila.")
        return {'rmse': np.nan, 'mae': np.nan}

    y_t = y_true.loc[common_idx]
    y_p = y_pred.loc[common_idx]

    # CRITICAL FIX: Remove NaNs
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]

    if len(y_t) == 0:
        print(f"Warning: {model_name} predictions mein sirf NaNs thay.")
        return {'rmse': np.nan, 'mae': np.nan}

    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    
    print(f"[{model_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {'rmse': rmse, 'mae': mae}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_start', type=str, default='2019-01-01')
    parser.add_argument('--skip_lstm', action='store_true')
    args = parser.parse_args()
    
    # ------------------------------------------------------
    # STEP 1: LOAD DATA & FEATURES
    # ------------------------------------------------------
    print("\n=== Step 1: Loading Data & Features ===")
    loader = DataLoader()
    try:
        df = loader.load_data()
        print(f"Data Loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    # Check Professor's Features
    required = [
        'hour_of_the_day_sin', 'hour_of_the_day_cos',
        'day_of_the_week_sin', 'day_of_the_week_cos',
        'month_of_the_year_sin', 'month_of_the_year_cos'
    ]
    missing = [f for f in required if f not in df.columns]
    if missing:
        print(f"❌ Missing Features: {missing}")
    else:
        print("✅ All cyclical features (Sin/Cos) present.")

    target_col = 'GBP/mWh' if 'GBP/mWh' in df.columns else df.columns[0]
    
    test_start_dt = pd.to_datetime(args.test_start)
    train_df = df[df.index < test_start_dt].copy()
    test_df = df[df.index >= test_start_dt].copy()
    print(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")

    preds_df = pd.DataFrame(index=test_df.index)
    preds_df['Actual'] = test_df[target_col]
    metrics = {}

    # ------------------------------------------------------
    # STEP 2: NAIVE BASELINES
    # ------------------------------------------------------
    print("\n=== Step 2: Running Naive Baselines ===")
    naive = NaiveBaseline(df, target_col=target_col)
    
    # 24h
    p_24 = naive.predict_24h().reindex(test_df.index)
    metrics['Naive_24h'] = evaluate_robust(test_df[target_col], p_24, "Naive_24h")
    preds_df['Naive_24h'] = p_24
    
    # 7d
    p_7d = naive.predict_7d().reindex(test_df.index)
    metrics['Naive_7d'] = evaluate_robust(test_df[target_col], p_7d, "Naive_7d")
    preds_df['Naive_7d'] = p_7d

    # ------------------------------------------------------
    # STEP 3: XGBOOST
    # ------------------------------------------------------
    print("\n=== Step 3: Running XGBoost ===")
    try:
        xgb_model = XGBoostModel(df, target_col=target_col)
        xgb_model.preprocess()
        
        cutoff_idx = len(xgb_model.df[xgb_model.df.index < test_start_dt])
        
        # --- Hyperparameter Tuning ---
        print("Tuning XGBoost Hyperparameters (this may take a few minutes)...")
        best_params = xgb_model.tune_hyperparameters(train_end_idx=cutoff_idx, n_iter=10)
        
        print(f"Training XGBoost with best params: {best_params}")
        xgb_model.train(train_end_idx=cutoff_idx, params=best_params)
        
        y_pred = xgb_model.predict(test_start_idx=cutoff_idx)
        
        # Alignment
        pred_timestamps = xgb_model.df.index[cutoff_idx:] + pd.Timedelta(hours=24)
        xgb_series = pd.Series(y_pred, index=pred_timestamps).reindex(test_df.index)
        
        metrics['XGBoost'] = evaluate_robust(test_df[target_col], xgb_series, "XGBoost")
        preds_df['XGBoost'] = xgb_series
    except Exception as e:
        print(f"XGBoost Error: {e}")

    # ------------------------------------------------------
    # STEP 4: LSTM (Reduced Epochs)
    # ------------------------------------------------------
    if not args.skip_lstm:
        print("\n=== Step 4: Running LSTM ===")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
            import tensorflow as tf
            tf.random.set_seed(SEED)
            from LSTM.lstm_model import LSTMModel
            
            lstm = LSTMModel(train_df, target_col=target_col, window_size=168, forecast_horizon=24)
            
            # --- CHANGE HERE: Increased Epochs & Complexity ---
            print("Training LSTM (Epochs=50, tuned config)...")
            lstm.train(
                epochs=50, 
                batch_size=32,
                units_1=128,    # Increased capacity
                units_2=64,
                dropout=0.3,    # Higher dropout for regularization
                learning_rate=0.001
            )
            
            # Predict
            combined_df = pd.concat([train_df.iloc[-168:], test_df])
            y_pred_raw, _ = lstm.predict_on_df(combined_df)
            
            # Alignment
            forecast_24h = y_pred_raw[:, -1]
            prediction_times = combined_df.index[168 : 168+len(forecast_24h)]
            
            lstm_series = pd.Series(forecast_24h, index=prediction_times)
            lstm_series.index = lstm_series.index + pd.Timedelta(hours=24)
            lstm_final = lstm_series.reindex(test_df.index)
            
            metrics['LSTM'] = evaluate_robust(test_df[target_col], lstm_final, "LSTM")
            preds_df['LSTM'] = lstm_final
            
        except ImportError:
            print("TensorFlow not installed.")
        except Exception as e:
            print(f"LSTM Error: {e}")
            traceback.print_exc()
    else:
        print("Skipping LSTM.")

    # ------------------------------------------------------
    # STEP 5: SAVE RESULTS
    # ------------------------------------------------------
    print("\n=== Final Summary ===")
    with open("model_results_summary.txt", 'w') as f:
        f.write("Model Results\n=============\n")
        for m, vals in metrics.items():
            if not pd.isna(vals['rmse']):
                line = f"{m}: RMSE={vals['rmse']:.4f}, MAE={vals['mae']:.4f}"
                print(line)
                f.write(line + "\n")

    print("\nGenerating Plot...")
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    subset = preds_df.last('14D')
    plt.figure(figsize=(15, 7))
    plt.plot(subset.index, subset['Actual'], 'k-', label='Actual', lw=2)
    if 'Naive_24h' in subset: plt.plot(subset.index, subset['Naive_24h'], '--', label='Naive 24h')
    if 'Naive_7d' in subset: plt.plot(subset.index, subset['Naive_7d'], ':', label='Naive 7d')
    if 'XGBoost' in subset: plt.plot(subset.index, subset['XGBoost'], label='XGBoost')
    if 'LSTM' in subset: plt.plot(subset.index, subset['LSTM'], 'r-', label='LSTM')

    plt.title('Forecast Comparison (Last 14 Days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'run_forecast_comparison.png'))
    print(f"Plot saved to plots/run_forecast_comparison.png")

if __name__ == "__main__":
    main()