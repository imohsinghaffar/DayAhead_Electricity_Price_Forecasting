# 📊 Pipeline Step-by-Step Guide

## Step 1: Run the Pipeline

```bash
cd "/Users/mohsinghaffar/Documents/University Data/Winter 2025 - 2026/Day Ahead EPF/ElectricityPricePrediction"
python run_forecast.py --visualize_optuna
```

---

## Step 2: What Happens (Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE EXECUTION FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: DATA LOADING
════════════════════
    data/
    ├── GUI_ENERGY_PRICES_2024.csv  ─────┐
    ├── CoalPrices.csv              ─────┼──▶ DataLoader2024 ──▶ DataFrame
    └── GasPrice.csv                ─────┘     (8,616 rows x 28 features)
    
                            │
                            ▼
                            
STEP 2: TRAIN/TEST SPLIT
═══════════════════════
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Jan ─────────────── Oct 21 ────────────────────────────────── Dec  │
    │  ◀─────── TRAIN (80%) ──────▶│◀────────── TEST (20%) ─────────────▶│
    │       (6,892 samples)        │         (1,724 samples)              │
    └─────────────────────────────────────────────────────────────────────┘
    
                            │
                            ▼

STEP 3: MODEL TRAINING
═══════════════════════

    ╔═══════════════════╗     ╔═══════════════════╗     ╔═══════════════════╗
    ║   NAIVE (24h)     ║     ║     XGBoost       ║     ║       LSTM        ║
    ╠═══════════════════╣     ╠═══════════════════╣     ╠═══════════════════╣
    ║ • No training     ║     ║ • 181 trees       ║     ║ • 3 layers        ║
    ║ • Uses lag-24     ║     ║ • All 27 features ║     ║ • 50 epochs max   ║
    ║ • Simple baseline ║     ║ • Gradient boost  ║     ║ • Early stopping  ║
    ╚═══════════════════╝     ╚═══════════════════╝     ╚═══════════════════╝
           │                         │                         │
           └─────────────────────────┴─────────────────────────┘
                                     │
                                     ▼

STEP 4: PREDICTION & EVALUATION
════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Model              │  RMSE      │  MAE       │  Performance       │
    ├─────────────────────┼────────────┼────────────┼────────────────────┤
    │  Naive (24h)        │  66.12     │  37.99     │  ████████ Baseline │
    │  XGBoost            │  66.21     │  35.43     │  ████████ Best MAE │
    │  LSTM               │  77.09     │  42.98     │  ██████ Needs work │
    └─────────────────────────────────────────────────────────────────────┘
    
                            │
                            ▼

STEP 5: VISUALIZATIONS
══════════════════════

    results/plots/
    ├── forecast_last14days.png    ──▶ Actual vs Predicted
    ├── metrics_rmse_mae.png       ──▶ Bar chart comparison
    ├── feature_importance.png     ──▶ XGBoost features
    └── error_distribution.png     ──▶ Residual histogram
    
    training_logs/
    ├── lstm_training_*.png        ──▶ Training curves
    └── optuna_plots/              ──▶ Hyperparameter analysis
    
    Analysis/plots/
    ├── comprehensive_analysis.png ──▶ 4-panel summary
    ├── feature_impact_detailed.png──▶ Feature importance
    └── forecasting_insights.png   ──▶ Error patterns
```

---

## Step 3: Check Results

### Quick Metrics
```
cat results/analysis_results_final.txt
```

### Training History
```
open training_logs/lstm_training_curves_*.png
```

### All Plots
```
open Analysis/plots/comprehensive_analysis.png
```

---

## File Locations Summary

| What You Need | Where to Find It |
|---------------|------------------|
| **Run pipeline** | `python run_forecast.py` |
| **See metrics** | `results/analysis_results_final.txt` |
| **Model predictions** | `results/forecast_values_2024.csv` |
| **LSTM training curves** | `training_logs/lstm_training_*.png` |
| **Optuna hyperparameters** | `training_logs/optuna_plots/` |
| **Comprehensive analysis** | `Analysis/plots/comprehensive_analysis.png` |
| **Training stats (JSON)** | `Analysis/training_stats_latest.json` |

---

## Common Commands

```bash
# View quick results
cat results/analysis_results_final.txt

# Open training curves
open training_logs/lstm_training_curves_*.png

# Open all analysis plots
open Analysis/plots/*.png

# Run with more data (better results)
python run_forecast.py --use_historical --visualize_optuna
```
