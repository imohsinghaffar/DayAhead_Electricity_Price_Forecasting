import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix for Mac segfault
import matplotlib.pyplot as plt
import os
import sys
import subprocess
from utils.data_loader import DataLoader
from models.xgboost_model import XGBoostModel

# Safety check for XGBoost
def is_module_safe(module_name):
    try:
        subprocess.check_call([sys.executable, "-c", f"import {module_name}"], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
        return True
    except:
        return False

XGBOOST_AVAILABLE = is_module_safe("xgboost")

def generate_dummy_data(days=365):
    """Generates dummy data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='H')
    price = np.random.normal(50, 10, size=len(dates)) + np.sin(np.linspace(0, 100, len(dates))) * 20
    load = np.random.normal(1000, 100, size=len(dates)) + np.cos(np.linspace(0, 100, len(dates))) * 200
    df = pd.DataFrame({'Price': price, 'LoadForecast': load}, index=dates)
    return df

def main():
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available on this system. Cannot run impact analysis.")
        return

    print("Step 1: Loading Data...")
    loader = DataLoader()
    try:
        df = loader.load_data()
    except FileNotFoundError:
        print("Data files not found. Using dummy data for demonstration.")
        df = generate_dummy_data()

    # Split data
    train_size = int(len(df) * 0.8)
    test_start_idx = train_size
    
    # Scenario 1: Price Only
    print("\nStep 2: Training XGBoost (Price Only)...")
    # Create a dataframe with Price and a Lag feature, but NO LoadForecast
    df_price_only = df[['Price']].copy()
    # Add a lag feature so the model has something to predict from (autoregression)
    # XGBoostModel doesn't automatically create lags, it expects features.
    df_price_only['Price_Lag24'] = df_price_only['Price'].shift(24)
    df_price_only.dropna(inplace=True)
    
    # We need to align indices after dropna
    # But wait, XGBoostModel splits by index. If we drop rows, indices change/shift relative to original df.
    # Better to keep the dataframe structure but just drop the LoadForecast column.
    
    df_price_only = df.drop(columns=['LoadForecast']).copy()
    # Ensure we have at least one feature besides target. 
    # If original df only had Price and LoadForecast, dropping LoadForecast leaves only Price.
    # XGBoostModel removes target from features -> 0 features.
    # So we MUST add a feature.
    df_price_only['Price_Lag24'] = df_price_only['Price'].shift(24)
    df_price_only.fillna(method='ffill', inplace=True) # Handle NaN from shift
    df_price_only.fillna(method='bfill', inplace=True) # Handle initial NaNs

    xgb_price = XGBoostModel(df_price_only, target_col='Price')
    xgb_price.preprocess()
    xgb_price.train(train_end_idx=train_size)
    metrics_price = xgb_price.evaluate(test_start_idx=test_start_idx)
    print(f"Price Only: RMSE={metrics_price['rmse']:.4f}, MAE={metrics_price['mae']:.4f}")

    # Scenario 2: Price + Load Forecast
    print("\nStep 3: Training XGBoost (Price + Load Forecast)...")
    # Add lag to full df too for fair comparison
    df_full = df.copy()
    df_full['Price_Lag24'] = df_full['Price'].shift(24)
    df_full.fillna(method='ffill', inplace=True)
    df_full.fillna(method='bfill', inplace=True)
    
    xgb_full = XGBoostModel(df_full, target_col='Price')
    xgb_full.preprocess()
    xgb_full.train(train_end_idx=train_size)
    metrics_full = xgb_full.evaluate(test_start_idx=test_start_idx)
    print(f"With Load:  RMSE={metrics_full['rmse']:.4f}, MAE={metrics_full['mae']:.4f}")

    # Plotting
    print("\nStep 4: Generating Impact Plot...")
    plt.figure(figsize=(15, 8))
    plot_len = 240
    if len(df) > plot_len:
        plot_idx = df.index[-plot_len:]
        plt.plot(plot_idx, df['Price'].iloc[-plot_len:], label='Actual', color='black', linewidth=2)
        
        if len(metrics_price['y_pred']) >= plot_len:
            plt.plot(plot_idx, metrics_price['y_pred'][-plot_len:], label='Price Only', linestyle='--')
            
        if len(metrics_full['y_pred']) >= plot_len:
            plt.plot(plot_idx, metrics_full['y_pred'][-plot_len:], label='Price + Load', color='green')

    plt.title('Impact of Load Forecast on Prediction Accuracy (XGBoost)')
    plt.xlabel('Time')
    plt.ylabel('Price (GBP/mWh)')
    plt.legend()
    plt.grid(True)
    
    plot_path = 'ElectricityPricePrediction/impact_analysis_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
