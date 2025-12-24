import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust paths
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader_2024 import DataLoader2024
from Naive.naive_baseline import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel
from LSTM.lstm_model import LSTMModel

# Set style
plt.style.use('ggplot')

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Forecast Runner")
    parser.add_argument("--test_start", type=str, help="Start date for test set (YYYY-MM-DD)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study on features")
    parser.add_argument("--tune_xgb", action="store_true", help="Run active Optuna tuning for XGBoost")
    return parser.parse_args()

def evaluate_preds(y_true, y_pred, model_name="Model"):
    # robust alignment using concat
    try:
        df_eval = pd.concat([y_true, y_pred], axis=1).dropna()
        df_eval.columns = ['Actual', 'Predicted']
    except Exception as e:
        print(f"Eval Error ({model_name}): {e}")
        return np.nan, np.nan
        
    if len(df_eval) == 0:
        print(f"Eval Warning ({model_name}): Empty intersection. y_true len={len(y_true)}, y_pred len={len(y_pred)}")
        return np.nan, np.nan
        
    rmse = np.sqrt(np.mean((df_eval['Actual'] - df_eval['Predicted'])**2))
    mae = np.mean(np.abs(df_eval['Actual'] - df_eval['Predicted']))
    return rmse, mae

def run_ablation(df, target_col='Price'):
    print("\n=== Running Ablation Study (XGBoost) ===")
    
    # Define Feature Sets
    # Base: Cyclical
    base_feats = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    
    # Sets to test
    experiments = {
        'Cyclical Only': base_feats,
        '+Weekly Load': base_feats + ['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast'],
        '+Fuels': base_feats + ['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast', 'Coal_Price', 'Gas_Price'],
        '+Load Forecast (All)': [c for c in df.columns if c != target_col]
    }
    
    results = []
    
    # Split for ablation (80/20)
    split_idx_raw = int(len(df) * 0.8)
    split_date = df.index[split_idx_raw]
    
    for name, feats in experiments.items():
        print(f"Testing Feature Set: {name}")
        
        xgb_model = XGBoostModel(df, target_col=target_col, feature_cols=feats)
        xgb_model.preprocess() 
        
        # Find correct train_end_idx in PREPROCESSED frame
        if hasattr(xgb_model, 'valid_indices'):
            train_mask = xgb_model.valid_indices < split_date
            train_end_idx = sum(train_mask)
        else:
            # Fallback if preprocessing failed to create valid_indices (should not happen)
            train_end_idx = split_idx_raw
            
        if train_end_idx == 0:
            print("Warning: Training set empty after preprocessing. Skipping.")
            continue
        
        # Train
        xgb_model.train(train_end_idx=train_end_idx, verbose=False)
        
        # Evaluate
        # Use predict(date)
        xgb_preds = xgb_model.predict(start_date=split_date)
        y_true = df[target_col].loc[xgb_preds.index]
        
        rmse, mae = evaluate_preds(y_true, xgb_preds)
        
        results.append({
            'Experiment': name,
            'RMSE': rmse,
            'MAE': mae
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Ablation Results ---")
    print(res_df)
    res_df.to_csv('ablation_results.csv', index=False)
    print("Saved to ablation_results.csv")

def main():
    args = parse_args()
    
    # 1. Load Data
    loader = DataLoader2024()
    df = loader.load_data()
    
    target_col = 'Price'
    
    # 2. Determine Split
    if args.test_start:
        try:
            split_date = pd.to_datetime(args.test_start).tz_localize('UTC')
            # Find index
            train_mask = df.index < split_date
            split_idx = sum(train_mask)
        except:
            print(f"Could not parse split date {args.test_start}, using last 20%")
            split_idx = int(len(df) * 0.8)
            split_date = df.index[split_idx]
    else:
        # Default: Last 20%
        split_idx = int(len(df) * 0.8)
        split_date = df.index[split_idx]
        
    print(f"\nTest Split Date: {split_date}")
    
    if args.ablation:
        run_ablation(df, target_col)
        return
        
    # === Main Evaluation ===
    
    predictions = pd.DataFrame(index=df.index[split_idx:])
    
    # 1. Naive Baselines
    print("\n--- Running Naive Baselines ---")
    naive = NaiveBaseline(df, target_col=target_col)
    
    for method in ['persistence', 'naive_24h', 'naive_7d']:
        try:
            preds = naive.get_predictions(method)
            # Align to test set
            # preds index is full df index.
            # We want preds for timestamps >= split_date
            preds_test = preds.loc[predictions.index]
            predictions[f'Naive_{method}'] = preds_test
        except Exception as e:
            print(f"Naive {method} failed (possible sparse data): {e}")

    # 2. XGBoost
    print("\n--- Running XGBoost ---")
    xgb_model = XGBoostModel(df, target_col=target_col)
    xgb_model.preprocess()
    
    # Map split_date to train_end_idx
    if hasattr(xgb_model, 'valid_indices'):
        train_mask = xgb_model.valid_indices < split_date
        train_end_idx = sum(train_mask)
    else:
        train_end_idx = split_idx
        
    if train_end_idx > 0:
        if args.tune_xgb:
            print("Tuning XGBoost...")
            best_params = xgb_model.tune_hyperparameters(train_end_idx=train_end_idx, n_trials=10)
            xgb_model.train(train_end_idx=train_end_idx, params=best_params, verbose=True)
        else:
            xgb_model.train(train_end_idx=train_end_idx, verbose=True)
            
        xgb_preds = xgb_model.predict(start_date=split_date)
        # xgb_preds is Series with index.
        # Align to predictions dataframe
        # Reindex to ensure match
        predictions['XGBoost'] = xgb_preds.reindex(predictions.index)
    else:
        print("XGBoost training set empty (too much lag dropping?). Skipping.")
    
    # 3. LSTM
    print("\n--- Running LSTM ---")
    # Load params
    lstm_params_path = os.path.join(os.path.dirname(__file__), 'LSTM/best_lstm_params.json')
    if os.path.exists(lstm_params_path):
        with open(lstm_params_path, 'r') as f:
            lstm_params = json.load(f)
        print(f"Loaded LSTM Params: {lstm_params}")
    else:
        print("No LSTM params found, using defaults.")
        lstm_params = {}
        
    # Adjust window size if data is small (Verification Fix)
    window_size = 168
    if split_idx < window_size + 24:
        print("Warning: Small training set, reducing LSTM window to 24.")
        window_size = 24
        
    lstm = LSTMModel(df.iloc[:split_idx], target_col=target_col, window_size=window_size, forecast_horizon=24)
    # Train
    lstm.train(epochs=30, verbose=1, **lstm_params)
    
    # Predict on Test
    test_context = pd.concat([df.iloc[:split_idx].iloc[-window_size:], df.iloc[split_idx:]])
    try:
        y_pred_lstm_full, _ = lstm.predict_on_df(test_context)
        # y_pred_full is (N_test, 24).
        # Check output length.
        # predict_on_df returns predictions aligned with "valid windows" of test_context.
        # test_context = window + test.
        # It scans from window_size to End.
        # So outputs correspond to test part.
        
        y_pred_lstm_da = y_pred_lstm_full[:, -1]
        
        # Length check
        test_len = len(predictions)
        predictions['LSTM'] = np.nan # Initialize column
        
        if len(y_pred_lstm_da) == test_len:
             predictions['LSTM'] = y_pred_lstm_da
        else:
             print(f"LSTM output length mismatch: Got {len(y_pred_lstm_da)}, expected {test_len}")
             min_l = min(len(y_pred_lstm_da), test_len)
             predictions.iloc[:min_l, predictions.columns.get_loc('LSTM')] = y_pred_lstm_da[:min_l]

    except Exception as e:
        print(f"LSTM Prediction Failed: {e}")

    # === Evaluation & Plotting ===
    
    # Add Actuals
    predictions['Actual'] = df[target_col].iloc[split_idx:]
    
    # Drop rows where Actual or predictions are NaN (ensure common support)
    # predictions = predictions.dropna() # DON'T drop all, evaluation handles pair-wise
    
    print("\n--- Predictions Dataframe Info ---")
    print(predictions.info())
    print(predictions.head())
    print("----------------------------------")
    
    print("\n=== Final Metrics ===")
    metrics_list = []
    for col in predictions.columns:
        if col == 'Actual': continue
        rmse, mae = evaluate_preds(predictions['Actual'], predictions[col], model_name=col)
        metrics_list.append({'Model': col, 'RMSE': rmse, 'MAE': mae})
        print(f"{col}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('results_metrics.csv', index=False)
    print("Saved results_metrics.csv")
    
    # Plot 1: Forecast Comparison (Last 14 days or full test)
    # If test is huge, plot last 14 days. If small, plot all.
    plot_len = min(len(predictions), 14*24)
    plot_df = predictions.iloc[-plot_len:]
    
    plt.figure(figsize=(15, 7))
    plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='black', linewidth=2, alpha=0.7)
    
    colors = {'Naive_persistence': 'gray', 'Naive_24h': 'silver', 'Naive_7d': 'lightgray', 
              'XGBoost': 'blue', 'LSTM': 'red'}
    
    for col in plot_df.columns:
        if col == 'Actual': continue
        style = '--' if 'Naive' in col else '-'
        width = 1 if 'Naive' in col else 2
        plt.plot(plot_df.index, plot_df[col], label=col, linestyle=style, linewidth=width, color=colors.get(col))
        
    plt.title('Forecast Comparison (Last 14 Days / Test Set)')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    
    os.makedirs('Analysis/plots', exist_ok=True)
    plt.savefig('Analysis/plots/forecast_last14days.png')
    print("Saved Analysis/plots/forecast_last14days.png")
    
    # Plot 2: Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    melted = metrics_df.melt(id_vars='Model', value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
    sns.barplot(data=melted, x='Model', y='Value', hue='Metric')
    plt.title('Model Evaluation: RMSE & MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Analysis/plots/metrics_rmse_mae.png')
    print("Saved Analysis/plots/metrics_rmse_mae.png")

if __name__ == "__main__":
    main()
