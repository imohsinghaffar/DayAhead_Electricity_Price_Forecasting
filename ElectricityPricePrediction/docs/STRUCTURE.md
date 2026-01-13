# Project Structure & Workflow (Updated)

## 📁 New Organized Directory Structure

```
ElectricityPricePrediction/
│
├── 📊 data/                          # Raw data files (unchanged)
│   ├── GUI_ENERGY_PRICES_2024.csv   # Main price data (Sequence 1)
│   ├── CoalPrices.csv               # Monthly coal prices
│   ├── MonthlyGasPrice.csv          # Monthly gas prices
│   ├── LOAD_DAYAHEAD_FullYear_Data.csv  # Weekly load forecasts
│   └── [historical data files...]   # 2013-2019 data, other fuels
│
├── 🔧 utils/                         # Data processing utilities
│   ├── data_loader_2024.py          # Enhanced data loader (28 features!)
│   ├── training_logger.py           # NEW: Training statistics logger
│   └── __init__.py
│
├── 🤖 models/                        # NEW: All models grouped together
│   ├── Naive/                       # Baseline models
│   │   ├── naive_baseline.py        # Persistence & seasonal baselines
│   │   └── __init__.py
│   │
│   ├── XGBoost/                     # Gradient boosting model
│   │   ├── xgboost_model.py         # Model implementation
│   │   ├── optuna_trial_history.json  # 🔑 Hyperparameter tuning results
│   │   └── __init__.py
│   │
│   └── LSTM/                        # Deep learning model (3-layer!)
│       ├── lstm_model.py            # Improved LSTM architecture
│       ├── lstm_optuna.py           # Hyperparameter optimization
│       ├── best_lstm_params.json    # 🔑 Best parameters found
│       ├── optuna_trial_history.json  # 🔑 All trial results
│       └── __init__.py
│
├── 📊 results/                      # NEW: All outputs organized
│   ├── plots/                       # All generated visualizations
│   │   ├── forecast_last14days.png
│   │   ├── metrics_rmse_mae.png
│   │   ├── feature_importance.png
│   │   └── error_distribution.png
│   │
│   ├── csv/                         # All CSV results
│   │   ├── results_metrics.csv
│   │   └── ablation_results.csv
│   │
│   ├── forecast_values_2024.csv     # Detailed hour-by-hour predictions
│   └── analysis_results_final.txt   # Summary report
│
├── 📝 training_logs/                # NEW: Training history JSONs
│   ├── lstm_training_YYYYMMDD_HHMMSS.json
│   ├── xgboost_training_YYYYMMDD_HHMMSS.json
│   ├── lstm_latest.json
│   └── xgboost_latest.json
│
├── 📁 legacy/                       # Old/reference code (preserved)
│   ├── forecast_next_day.py
│   ├── impact_analysis.py
│   └── run_2024_analysis.py
│
├── 📚 docs/                         # NEW: Documentation
│   ├── STRUCTURE.md                 # This file
│   └── README.md                    # Project overview
│
├── 🚀 run_forecast.py               # MAIN PIPELINE SCRIPT
├── 📄 LOAD_DAYAHEAD_FullYear_Data.csv  # Weekly load data
├── 📄 requirements.txt              # Python dependencies
└── 📄 .gitignore                    # Git ignore rules
```

## 🔄 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    1. DATA LOADING                          │
│  utils/data_loader_2024.py (ENHANCED!)                      │
│  • Load GUI_ENERGY_PRICES_2024.csv (Sequence 1)            │
│  • Merge fuel prices (Coal, Gas)                           │
│  • Add cyclical features (hour_sin/cos, day, month)        │
│  • NEW: Add lag features (1h, 24h, 48h, 168h)             │
│  • NEW: Add momentum indicators (volatility, changes)      │
│  • NEW: Add time indicators (weekend, peak hours)          │
│  • Result: 28 features (was 12)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                2. TRAIN/TEST SPLIT                          │
│  • Training: Jan - Oct 19, 2024 (80%)                      │
│  • Testing:  Oct 19 - Dec 31, 2024 (20%)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                3. MODEL TRAINING                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Naive Models │  │   XGBoost    │  │ LSTM (NEW!)  │     │
│  │ • Lag 24h    │  │ • Optuna     │  │ • 3 layers   │     │
│  │ • Lag 168h   │  │   tuning     │  │ • 128→64→32  │     │
│  │              │  │ • Enhanced   │  │ • Optuna     │     │
│  │              │  │   features   │  │   tuning     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  Training stats saved to training_logs/ as JSON            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                4. PREDICTION                                │
│  All models predict on same test set (Oct-Dec 2024)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                5. EVALUATION & LOGGING                      │
│  • Calculate RMSE, MAE for each model                      │
│  • Generate comparison plots                               │
│  • Save results to results/csv/                            │
│  • Save training logs to training_logs/                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                6. OUTPUT FILES                              │
│  • results/csv/results_metrics.csv                         │
│  • results/forecast_values_2024.csv                        │
│  • results/plots/*.png (4 plots)                           │
│  • training_logs/lstm_latest.json                          │
│  • training_logs/xgboost_latest.json                       │
│  • models/LSTM/optuna_trial_history.json                   │
│  • models/XGBoost/optuna_trial_history.json                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 How to Run

### Complete Pipeline
```bash
python run_forecast.py
```

### With Optuna Tuning (LSTM)
```bash
# Run LSTM optimization (20 trials)
python -m models.LSTM.lstm_optuna

# Then run full pipeline
python run_forecast.py
```

### Ablation Study
```bash
python run_forecast.py --ablation
```

## 📊 Key Improvements

### Enhanced Features (16 new!)
- **Lag Features**: price_lag_1, price_lag_24, price_lag_48, price_lag_168
- **Rolling Stats**: rolling_mean_24, rolling_std_24, rolling_min/max_24
- **Momentum**: price_diff_24, price_change_pct, volatility_24h/168h
- **Time Indicators**: is_weekend, is_peak_hour, is_business_hours

### Improved LSTM Architecture
- **Before**: 2 layers (64→32 units)
- **After**: 3 layers (128→64→32 units)
- Deeper dense layers (128→64)
- Better gradient clipping

### Training Transparency
- All training runs logged to `training_logs/`
- JSON files include:
  - Training history (loss curves, epochs)
  - Model configuration (all hyperparameters)
  - Performance metrics (RMSE, MAE, improvement %)
  - Timestamps

## 📁 Important Files

### 🔑 JSON Result Files (DO NOT DELETE)
- `models/LSTM/best_lstm_params.json` - Best LSTM hyperparameters
- `models/LSTM/optuna_trial_history.json` - All 20 Optuna trials
- `models/XGBoost/optuna_trial_history.json` - XGBoost trials
- `training_logs/*.json` - Training history for each run

### 📊 CSV Result Files
- `results/csv/results_metrics.csv` - Final RMSE/MAE for all models
- `results/csv/ablation_results.csv` - Feature importance study
- `results/forecast_values_2024.csv` - Hour-by-hour predictions

### 📈 Visualization Files
- `results/plots/forecast_last14days.png` - Time series comparison
- `results/plots/metrics_rmse_mae.png` - Model performance bars
- `results/plots/feature_importance.png` - XGBoost feature ranking
- `results/plots/error_distribution.png` - Residual analysis

## 🎯 Expected Performance

After improvements, target performance hierarchy:
```
Naive Baseline (24h):  ~90 RMSE  (baseline)
XGBoost:              ~60-65 RMSE  (30-35% better)
LSTM (3-layer):       ~50-55 RMSE  (40-45% better)
```

## 📝 Notes

- All JSON files contain timestamps for tracking experiments
- Legacy code preserved in `legacy/` for reference
- All plots regenerated on each run for consistency
- Training logs accumulate over time for improvement tracking
- Enhanced features should significantly improve model performance
