import pandas as pd
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader_2024 import DataLoader2024
from Naive.naive_baselines import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel
from LSTM.lstm_model import LSTMModel

def predict_next_day(lstm_params=None):
    print("=== Forecasting Next Day Prices ===")
    
    # 1. Load Data (Up to current moment)
    loader = DataLoader2024()
    df = loader.load_data()
    print(f"Latest Data Point: {df.index[-1]}")
    
    # 2. XGBoost Forecast
    print("\n--- Running XGBoost ---")
    xgb = XGBoostModel(df, target_col='Price')
    xgb.preprocess()
    # Train on EVERYTHING
    xgb.train(train_end_idx=len(xgb.X_scaled), verbose=False) # Train on full data
    
    # For XGBoost, prediction for T+24 requires input at T.
    # The last available row in df corresponds to T.
    # So we predict T+24h.
    last_X = xgb.X_scaled[-1:]
    xgb_pred_scaled = xgb.model.predict(last_X)
    xgb_price_next_24h = xgb.scaler_y.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).item()
    
    # Wait, the current model predicts *one* step 24h ahead? 
    # Or does it predict the *sequence* of the next day?
    # Our model target is `shift(-24)`. 
    # Row[t] has features[t] and target price[t+24].
    # So using Row[last], we predict Price[last+24h].
    # But for a full day forecast (hours 0-23 of tomorrow), we need row inputs for hours (last-23) to (last).
    
    # Let's predict the next 24 hours.
    # We need the last 24 rows of features.
    last_24_X = xgb.X_scaled[-24:]
    xgb_preds_scaled = xgb.model.predict(last_24_X)
    xgb_forecast = xgb.scaler_y.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).flatten()
    
    # Timestamps for tomorrow
    last_time = df.index[-1]
    future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(24)]
    # Actually, if row[t] predicts t+24.
    # row[last] predicts last+24h.
    # row[last-23] predicts (last-23)+24 = last+1h.
    # Yes, so taking last 24 rows gives next 24h forecast.
    # Timestamps start from last_time + 1h
    
    # 3. LSTM Forecast
    print("\n--- Running LSTM ---")
    # Using Best Known Params OR passed params
    if lstm_params is None:
        print("Using hardcoded default LSTM params...")
        params = {
            'units_1': 83, 'units_2': 19, 'dropout': 0.251, 'learning_rate': 0.003, 'batch_size': 64, 'epochs': 30
        }
    else:
        print(f"Using provided LSTM params: {lstm_params}")
        params = lstm_params
    
    # Train Residual Model
    naive = NaiveBaseline(df, target_col='Price')
    naive_preds = naive.predict_24h()
    df['Naive'] = naive_preds
    df['Residual'] = df['Price'] - df['Naive']
    df_clean = df.dropna(subset=['Residual'])
    
    lstm = LSTMModel(df_clean, target_col='Residual', window_size=168, forecast_horizon=24)
    lstm.train(**params)
    
    # Predict residual for next day
    # We need last sequence
    # Predict on df uses the sliding window properly
    # Using the last available window in df to predict future
    # We need to construct a 'future_df' specifically or use internal predict logic?
    # predict_on_df ends at the end of df.
    # To predict *beyond*, we usually feed the last window.
    
    # Extract last window
    # LSTM includes target in input_cols (autoregression)
    # So we must select lstm.input_cols, NOT lstm.feature_cols
    last_window_data = lstm.x_scaler.transform(df_clean[lstm.input_cols].values[-lstm.window_size:])
    last_window_data = last_window_data.reshape(1, lstm.window_size, len(lstm.input_cols))
    
    lstm_resid_pred_scaled = lstm.model.predict(last_window_data, verbose=0)
    # Fix attribute: y_scaler, not scaler_target
    lstm_resid_pred = lstm.y_scaler.inverse_transform(lstm_resid_pred_scaled).flatten()
    
    # Naive component for tomorrow
    # Naive(t+24) = Price(t)
    # The naive forecast for the next 24h corresponds to the Prices of the last 24h.
    naive_forecast = df['Price'].iloc[-24:].values
    
    lstm_forecast = naive_forecast + lstm_resid_pred
    
    # 4. Ensemble
    ensemble_forecast = (xgb_forecast + lstm_forecast) / 2
    
    # 5. Output
    forecast_df = pd.DataFrame({
        'Date': future_times,
        'Naive': naive_forecast,
        'XGBoost': xgb_forecast,
        'LSTM': lstm_forecast,
        'Ensemble': ensemble_forecast
    })
    
    print("\n=== Next Day Forecast ===")
    print(forecast_df)
    
    forecast_df.to_csv("Analysis/forecast_next_day.csv", index=False)
    print("Saved to Analysis/forecast_next_day.csv")
    
    # 6. Visualization
    print("\n--- Generating Forecast Plot ---")
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['Date'], forecast_df['Naive'], '--', label='Naive', alpha=0.7)
    plt.plot(forecast_df['Date'], forecast_df['XGBoost'], label='XGBoost', alpha=0.8)
    plt.plot(forecast_df['Date'], forecast_df['LSTM'], label='LSTM', alpha=0.8)
    plt.plot(forecast_df['Date'], forecast_df['Ensemble'], 'r-o', label='Ensemble', lw=2)
    
    plt.title(f'Next Day Electricity Price Forecast ({forecast_df["Date"].iloc[0].date()})')
    plt.ylabel('Price (EUR/MWh)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = "Analysis/plots/forecast_next_day.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    predict_next_day()
