import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import argparse
import warnings
import traceback
import random

# --- CONFIG ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

from utils.data_loader import DataLoader
from Naive.naive_baselines import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel

def evaluate_series(y_true, y_pred, model_name):
    # Robust Evaluate (No Crash on NaNs)
    common_idx = y_true.index.intersection(y_pred.index)
    if len(common_idx) == 0: return {"rmse": np.nan, "mae": np.nan}
    
    y_t = y_true.loc[common_idx]
    y_p = y_pred.loc[common_idx]
    
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[mask], y_p[mask]
    
    if len(y_t) == 0: return {"rmse": np.nan, "mae": np.nan}

    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    return {"rmse": rmse, "mae": mae}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_start', type=str, default='2019-01-01')
    parser.add_argument('--skip_lstm', action='store_true')
    args = parser.parse_args()

    # 1. Load Data
    print("--- Loading Data ---")
    loader = DataLoader()
    try:
        df = loader.load_data()
    except: return

    target_col = 'GBP/mWh' if 'GBP/mWh' in df.columns else df.columns[0]
    test_start_dt = pd.to_datetime(args.test_start)
    train_df = df[df.index < test_start_dt].copy()
    test_df  = df[df.index >= test_start_dt].copy()

    preds_df = pd.DataFrame(index=test_df.index)
    preds_df['Actual'] = test_df[target_col]
    metrics = {}

    # 2. Naive
    print("--- Naive ---")
    naive = NaiveBaseline(df, target_col=target_col)
    preds_df['Naive_24h'] = naive.predict_24h().reindex(test_df.index)
    metrics['Naive_24h'] = evaluate_series(test_df[target_col], preds_df['Naive_24h'], 'Naive_24h')
    
    preds_df['Naive_7d'] = naive.predict_7d().reindex(test_df.index)
    metrics['Naive_7d'] = evaluate_series(test_df[target_col], preds_df['Naive_7d'], 'Naive_7d')

    # 3. XGBoost
    print("--- XGBoost ---")
    try:
        xgb = XGBoostModel(df, target_col=target_col)
        xgb.preprocess()
        cutoff = len(xgb.df[xgb.df.index < test_start_dt])
        xgb.train(train_end_idx=cutoff)
        
        y_pred = xgb.predict(test_start_idx=cutoff)
        pred_dates = xgb.df.index[cutoff:] + pd.Timedelta(hours=24)
        
        preds_df['XGBoost'] = pd.Series(y_pred, index=pred_dates).reindex(test_df.index)
        metrics['XGBoost'] = evaluate_series(test_df[target_col], preds_df['XGBoost'], 'XGBoost')
    except Exception as e: print(f"XGB Error: {e}")

    # 4. LSTM
    if not args.skip_lstm:
        print("--- LSTM ---")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            import tensorflow as tf
            tf.random.set_seed(SEED)
            from LSTM.lstm_model import LSTMModel

            lstm = LSTMModel(train_df, target_col=target_col, window_size=168, forecast_horizon=24)
            
            # --- CHANGE HERE: Epochs reduced to 7 ---
            lstm.train(epochs=7, batch_size=32)
            
            combined = pd.concat([train_df.iloc[-168:], test_df])
            y_pred_arr, _ = lstm.predict_on_df(combined)
            
            forecasts = y_pred_arr[:, -1]
            pred_times = combined.index[168:168+len(forecasts)]
            
            lstm_series = pd.Series(forecasts, index=pred_times)
            lstm_series.index = lstm_series.index + pd.Timedelta(hours=24)
            
            preds_df['LSTM'] = lstm_series.reindex(test_df.index)
            metrics['LSTM'] = evaluate_series(test_df[target_col], preds_df['LSTM'], 'LSTM')
        except Exception as e: 
            print(f"LSTM Error: {e}")
            traceback.print_exc()

    # 5. Results
    print("\n=== RESULTS ===")
    for m, v in metrics.items():
        if not pd.isna(v['rmse']): print(f"{m}: RMSE={v['rmse']:.2f}, MAE={v['mae']:.2f}")

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    subset = preds_df.last('14D')
    plt.figure(figsize=(15, 7))
    plt.plot(subset.index, subset['Actual'], 'k-', label='Actual')
    if 'Naive_24h' in subset: plt.plot(subset.index, subset['Naive_24h'], '--', label='Naive 24h')
    if 'Naive_7d' in subset: plt.plot(subset.index, subset['Naive_7d'], ':', label='Naive 7d')
    if 'XGBoost' in subset: plt.plot(subset.index, subset['XGBoost'], label='XGBoost')
    if 'LSTM' in subset: plt.plot(subset.index, subset['LSTM'], 'r-', label='LSTM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'final_comparison.png'))
    print("Plot saved.")

if __name__ == "__main__":
    main()