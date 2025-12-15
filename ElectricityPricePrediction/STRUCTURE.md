# Project Structure

This project focuses on Day-Ahead Electricity Price Forecasting using XGBoost, LSTM, and Naive Baselines.

## Core Files

### 1. Main Execution
*   `Analysis/run_2024_analysis.py`: **The Master Script**.
    *   Loads data, trains all models (XGBoost, LSTM), creates the Ensemble, and generates all plots/metrics.
    *   Run this to get full results.

### 2. Models
*   `XGBoost/xgboost_model.py`: **XGBoost Regressor**.
    *   Handles data preprocessing, feature engineering (lags), and training with early stopping.
*   `LSTM/lstm_model.py`: **LSTM Neural Network**.
    *   TensorFlow/Keras implementation with residual learning (predicts error of Naive 24h).
*   `Naive/naive_baselines.py`: **Benchmarks**.
    *   Provides simple 24h and 7d persistence forecasts for comparison.

### 3. Data & Utilities
*   `utils/data_loader_2024.py`: **Data Pipeline**.
    *   Loads CSVs, fetches Weather data (OpenMeteo), and handles missing value interpolation.
*   `Analysis/run_optuna_enriched.py`: **Hyperparameter Tuning**.
    *   Automated script to find best LSTM parameters using Optuna.

### 4. Configuration
*   `requirements.txt`: List of all Python dependencies.

## Legacy
*   `legacy/`: Contains older scripts (`run_forecast.py`) and loaders that are no longer used.
