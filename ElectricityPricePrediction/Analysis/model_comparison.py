import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os
import argparse
import subprocess
import warnings
import traceback

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from Naive.naive_baselines import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel

# Try importing TensorFlow/LSTM
try:
    import tensorflow as tf
    from LSTM.lstm_model import LSTMModel
    tensorflow_available = True
    print("TensorFlow available. LSTM model will be included.")
except ImportError:
    tensorflow_available = False
    print("TensorFlow not available. LSTM model will be skipped.")

# Suppress warnings
warnings.filterwarnings("ignore")

def check_xgboost_availability():
    try:
        import xgboost
        return True
    except ImportError:
        return False

def load_data(filepath):
    """
    Loads the dataset.
    """
    if not os.path.exists(filepath):
        print(f"Data file not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
    return df

def generate_dummy_data():
    """
    Generates dummy data for testing if the real file is missing.
    """
    print("Generating dummy data...")
    dates = pd.date_range(start='2018-01-01', end='2019-01-31', freq='H')
    df = pd.DataFrame(index=dates)
    df['GBP/mWh'] = 50 + 10 * np.sin(np.arange(len(dates)) / 24) + np.random.normal(0, 5, len(dates))
    df['temperature'] = 10 + 5 * np.cos(np.arange(len(dates)) / 24)
    return df

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles division by zero by replacing 0 with a small epsilon or ignoring.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculates RMSE, MAE, and MAPE.
    """
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]

    if len(y_true) == 0:
        print(f"No common data points for {model_name}")
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    print(f"--- {model_name} Results ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return {'rmse': rmse, 'mae': mae, 'mape': mape}

def main():
    parser = argparse.ArgumentParser(description="Compare Naive, XGBoost, and LSTM models.")
    parser.add_argument('--data_path', type=str, default='data/re_fixed_multivariate_timeseires.csv', help='Path to the dataset')
    args = parser.parse_args()
    
    # 0. Load Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.data_path)
    df = load_data(data_path)
    
    if df is None:
        df = generate_dummy_data()

    # Ensure target column exists
    target_col = 'price' if 'price' in df.columns else 'GBP/mWh'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Using first column as target.")
        target_col = df.columns[0]
        
    print(f"Target Column: {target_col}")

    # 1. Define Split (Train: < 2019, Test: >= 2019)
    test_start_date = '2019-01-01'
    # Ensure index is sorted
    df.sort_index(inplace=True)
    
    train_df = df[df.index < test_start_date].copy()
    test_df = df[df.index >= test_start_date].copy()
    
    if test_df.empty:
        print(f"Test set is empty. Check data date range (Max date: {df.index.max()})")
        return

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    results = {}
    predictions = pd.DataFrame(index=test_df.index)
    predictions['Actual'] = test_df[target_col]

    # 2. Naive Baselines
    print("\nRunning Naive Baselines...")
    naive = NaiveBaseline(df, target_col=target_col)
    
    # Naive 24h
    pred_24h = naive.predict_24h()
    results['Naive_24h'] = evaluate_model(test_df[target_col], pred_24h, "Naive 24h")
    predictions['Naive_24h'] = pred_24h

    # Naive 7d
    pred_7d = naive.predict_7d()
    results['Naive_7d'] = evaluate_model(test_df[target_col], pred_7d, "Naive 7d")
    predictions['Naive_7d'] = pred_7d

    # 3. XGBoost
    if check_xgboost_availability():
        print("\nRunning XGBoost...")
        try:
            xgb_model = XGBoostModel(df, target_col=target_col)
            xgb_model.preprocess()
            
            # XGBoostModel uses integer indices on the internal scaled data
            train_size = len(train_df)
            xgb_model.train(train_end_idx=train_size) 
            
            xgb_pred_values = xgb_model.predict(test_start_idx=train_size)
            
            # Align with test_df index
            # Ensure length match
            min_len = min(len(xgb_pred_values), len(test_df))
            xgb_pred = pd.Series(xgb_pred_values[:min_len], index=test_df.index[:min_len])
            
            results['XGBoost'] = evaluate_model(test_df.loc[xgb_pred.index, target_col], xgb_pred, "XGBoost")
            predictions['XGBoost'] = xgb_pred
        except Exception as e:
            print(f"Error running XGBoost: {e}")
            traceback.print_exc()
    else:
        print("\nXGBoost not installed.")

    # 4. LSTM
    if tensorflow_available:
        print("\nRunning LSTM...")
        try:
            # Instantiate (7 Days Input = 168 hours)
            lstm = LSTMModel(train_df, target_col=target_col, window_size=168)
            
            # Preprocess (fit scaler)
            lstm.preprocess()
            
            # Train
            print("Training LSTM (epochs=5)...")
            lstm.train(epochs=5, batch_size=32) 
            
            # Predict
            window_size = lstm.window_size
            
            # Combine end of train with test to provide context
            # We need at least window_size points before test start
            combined_df = pd.concat([train_df.iloc[-window_size:], test_df])
            
            # Scale combined data using the SAME scaler from training
            data_to_scale = combined_df[lstm.feature_cols].values
            combined_scaled = lstm.scaler.transform(data_to_scale)
            
            # Predict
            lstm_preds_array = lstm.predict(data_scaled=combined_scaled)
            
            # Extract 24-hour ahead predictions (see logic below)
            # If shape is (N, 24), we take the last column (T+24 forecast made at T)
            # This aligns with "Day Ahead" prediction.
            
            if len(lstm_preds_array.shape) > 1 and lstm_preds_array.shape[1] >= 24:
                pred_series_values = lstm_preds_array[:, -1] # The last step (24th hour)
                
                # Align index:
                # The prediction made at input index `i` (which is `window_size` points ending at `i`)
                # is for target index `i + forecast_horizon` (roughly, strictly `i` to `i+hor`).
                # We want the LAST step of the horizon, so `i + forecast_horizon - 1`?
                # Actually, check create_sequences: y ends at `i + forecast_horizon`.
                # y is data[i : i+hor]. The last point is data[i+hor-1].
                # Input was data[i-window : i].
                # So if input ends at `i`, we predict up to `i+hor-1`.
                # In `combined_df`, index `i` is where test starts relative to combined (window_size).
                # So the first prediction corresponds to `window_size + 24 - 1` in combined_df.
                
                start_idx_offset = window_size + lstm.forecast_horizon - 1
                
                # Ensure we don't go out of bounds
                valid_length = min(len(pred_series_values), len(combined_df) - start_idx_offset)
                pred_series_values = pred_series_values[:valid_length]
                
                pred_index = combined_df.index[start_idx_offset : start_idx_offset + valid_length]
                
                lstm_pred_series = pd.Series(pred_series_values, index=pred_index)
                
                results['LSTM'] = evaluate_model(test_df[target_col], lstm_pred_series, "LSTM (24h ahead)")
                predictions['LSTM'] = lstm_pred_series
                
            else:
                 print(f"LSTM output shape {lstm_preds_array.shape} not supported for auto-alignment.")

        except Exception as e:
            print(f"Error running LSTM: {e}")
            traceback.print_exc()

    # Save metrics to file
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Model Comparison Metrics\n")
        f.write("========================\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  MAE:  {metrics['mae']:.2f}\n")
            f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
    print(f"\nMetrics saved to {metrics_path}")

    # 5. Plotting
    print("\nGenerating Comparison Plots...")
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Forecast vs Actual (Time Series)
    plt.figure(figsize=(15, 7))
    plot_days = 14
    subset = predictions.last(f'{plot_days}D')
    
    plt.plot(subset.index, subset['Actual'], label='Actual', color='black', alpha=0.7, linewidth=2)
    
    if 'Naive_24h' in subset.columns:
        plt.plot(subset.index, subset['Naive_24h'], label='Naive 24h', linestyle='--')
    
    if 'XGBoost' in subset.columns:
        plt.plot(subset.index, subset['XGBoost'], label='XGBoost')

    if 'LSTM' in subset.columns:
        plt.plot(subset.index, subset['LSTM'], label='LSTM', color='red')

    plt.title(f'Forecast Comparison (Last {plot_days} Days) - 7 Day Input / 24h Forecast')
    plt.ylabel(target_col)
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'comparison_plot.png'))
    print(f"Forecast plot saved to {os.path.join(plot_dir, 'comparison_plot.png')}")

    # Plot 2: Metrics Comparison (Bar Chart)
    # Prepare data for plotting
    models = list(results.keys())
    rmses = [results[m]['rmse'] for m in models]
    maes = [results[m]['mae'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rmses, width, label='RMSE', color='skyblue')
    rects2 = ax.bar(x + width/2, maes, width, label='MAE', color='lightgreen')

    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance Comparison (RMSE & MAE)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metrics_plot.png'))
    print(f"Metrics bar chart saved to {os.path.join(plot_dir, 'metrics_plot.png')}")

if __name__ == "__main__":
    main()
