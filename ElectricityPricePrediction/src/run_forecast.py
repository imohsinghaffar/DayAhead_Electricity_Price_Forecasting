#!/usr/bin/env python3
"""
================================================================================
                    ELECTRICITY PRICE FORECASTING PIPELINE
================================================================================
Main entry point for the electricity price prediction system.
Supports Naive, XGBoost, and Probabilistic LSTM models with German weather data.

Requirements:
- German weather data (DWD API)
- Fuel prices (Oil, Coal, Gas)
- Historical data (2019-2024)
- Probabilistic forecasting (Uncertainty)
================================================================================
"""

import argparse
import os
import sys
import json
import logging
import random
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set Seed for Reproducibility
# Standardized Seeding
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Tensorflow seeding
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    # XGBoost seeding is handled in the model class

def setup_directories(project_root):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = project_root / "Analysis"
    latest_dir = analysis_dir / "Latest"
    
    # 1. Auto-archive previous Latest to Training_N
    if latest_dir.exists() and any(latest_dir.iterdir()):
        # Find next Training number
        existing_training = [d.name for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith("Training_")]
        if existing_training:
            max_num = max([int(d.split("_")[1]) for d in existing_training])
            next_num = max_num + 1
        else:
            next_num = 1
        
        archive_dir = analysis_dir / f"Training_{next_num}"
        shutil.move(str(latest_dir), str(archive_dir))
        logger.info(f"Archived previous run to Training_{next_num}")
    
    # 2. Create fresh Latest folder
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Create Latest Subfolders
    for d in ["Plots/Forecasting", "Plots/Error_Analysis", "Plots/Features", "Plots/Interactive"]:
        (latest_dir / d).mkdir(parents=True, exist_ok=True)
    
    return run_id, latest_dir, analysis_dir


# Model name to technique-based folder mapping
MODEL_MAP = {
    "LSTM": "LSTM_Residual_Learning",
    "XGBoost": "XGBoost_Weather_Tuned",
    "XGB_LagCyclical": "XGBoost_LagCyclical",
    "Naive_persistence": "Naive_Persistence",
    "Naive_naive_24h": "Naive_24h",
    "Naive_naive_7d": "Naive_7d",
    "XGBoost_Tuning": "XGBoost_Weather_Tuned/Optuna"
}

def save_model_result(analysis_dir, model_name, data):
    """Saves a model-specific JSON result in Models/<TechniqueName>/Results/."""
    technique_name = MODEL_MAP.get(model_name, model_name)
    results_dir = analysis_dir / "Models" / technique_name / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get next sequential number for this specific model
    existing = list(results_dir.glob("*.json"))
    next_num = len(existing) + 1
    
    filename = f"{next_num}.json"
    with open(results_dir / filename, 'w') as f:
        json.dump(data, f, indent=4)
        
    logger.info(f"Saved {model_name} result to Models/{technique_name}/Results/{filename}")
    return next_num

def evaluate_preds(actual, pred, model_name, baseline_rmse=None):
    valid = ~(np.isnan(actual) | np.isnan(pred))
    n_points = valid.sum()
    if n_points == 0:
        logger.warning(f"Empty intersection for {model_name}. Skipping evaluation.")
        return np.nan, np.nan, 0, np.nan, np.nan
        
    a, p = actual[valid], pred[valid]
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    rmse = np.sqrt(mean_squared_error(a, p))
    mae = mean_absolute_error(a, p)
    
    delta_rmse = rmse - baseline_rmse if baseline_rmse is not None else np.nan
    delta_pct = (delta_rmse / baseline_rmse * 100) if baseline_rmse and baseline_rmse != 0 else np.nan
    
    return rmse, mae, n_points, delta_rmse, delta_pct

# Adjust paths to support execution from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from utils.data_loader_2024 import DataLoader2024
    from utils.historical_data_loader import HistoricalDataLoader
    from utils.weather_data_loader import WeatherDataLoader, merge_weather_with_prices
    from utils.enhanced_visualizations import (
        create_comprehensive_comparison,
        create_feature_impact_analysis,
        create_forecasting_insights
    )
    from utils.optuna_visualizer import visualize_optuna_history
    
    from models.Naive.naive_baseline import NaiveBaseline
    from models.XGBoost.xgboost_model import XGBoostModel
    from models.LSTM.lstm_model import LSTMModel
    from models.LSTM.probabilistic_lstm import ProbabilisticLSTM
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

# Set style
plt.style.use('ggplot')

def parse_args():
    parser = argparse.ArgumentParser(description="Electricity Price Forecast Runner (A-Z Refactor)")
    parser.add_argument("--test_start", type=str, help="Start date for test set (YYYY-MM-DD)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study on features")
    parser.add_argument("--tune_xgb", action="store_true", help="Run Optuna tuning for XGBoost")
    parser.add_argument("--tune_lstm", action="store_true", help="Run Optuna tuning for LSTM")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for tuning")
    parser.add_argument("--use_historical", action="store_true", help="Load 2019-2024 historical data")
    parser.add_argument("--use_weather", action="store_true", help="Include German DWD weather data")
    parser.add_argument("--probabilistic", action="store_true", help="Use Probabilistic LSTM (Uncertainty)")
    parser.add_argument("--visualize_optuna", action="store_true", help="Generate Optuna plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def run_ablation(df, target_col='Price'):
    logger.info("Starting Feature Ablation Study...")
    
    # Professor-defined cyclical features
    base_feats = [
        'hour_of_the_day_sin', 'hour_of_the_day_cos', 
        'day_of_the_week_sin', 'day_of_the_week_cos', 
        'month_of_the_year_sin', 'month_of_the_year_cos'
    ]
    
    experiments = {
        'Cyclical Only': base_feats,
        '+Weekly Load': base_feats + ['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast'],
        '+Fuels': base_feats + ['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast', 'Coal_Price', 'Gas_Price'],
        '+Full Model': [c for c in df.columns if c != target_col]
    }
    
    results = []
    split_date = df.index[int(len(df) * 0.8)]
    
    for name, feats in experiments.items():
        logger.info(f"Testing Experiment: {name}")
        # Only use features that exist in df
        valid_feats = [f for f in feats if f in df.columns]
        
        xgb_model = XGBoostModel(df, target_col=target_col, feature_cols=valid_feats)
        xgb_model.preprocess()
        
        split_mask = xgb_model.valid_indices < split_date
        train_end_idx = sum(split_mask)
        
        xgb_model.train(train_end_idx=train_end_idx, verbose=False)
        xgb_preds = xgb_model.predict(start_date=split_date)
        
        y_true = df[target_col].loc[xgb_preds.index]
        rmse, mae, _, _, _ = evaluate_preds(y_true, xgb_preds, name)
        
        results.append({'Experiment': name, 'RMSE': rmse, 'MAE': mae})
        
    res_df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "results" / "csv" / "ablation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    logger.info(f"Ablation results saved to: {output_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Paths setup
    PROJECT_ROOT = Path(__file__).parent.parent
    RUN_ID, LATEST_DIR, LEGACY_DIR = setup_directories(PROJECT_ROOT)
    
    # Logging Setup
    log_file = LATEST_DIR / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Run: {RUN_ID}")
    logger.info(f"Global seed set to: {args.seed}")
    
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = LATEST_DIR  # Standardize all results to Analysis/latest
    
    # 1. Load Data
    if args.use_historical:
        logger.info("Loading Historical Dataset (2019-2024)...")
        from utils.historical_data_loader import HistoricalDataLoader
        loader = HistoricalDataLoader()
        df = loader.load_full_dataset()
    else:
        logger.info("Loading 2024 Dataset...")
        from utils.data_loader_2024 import DataLoader2024
        loader = DataLoader2024()
        df = loader.load_data()
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return

    target_col = 'Price'
    
    # 2. Weather Integration
    if args.use_weather:
        logger.info("Integrating German DWD Weather Data...")
        from utils.weather_data_loader import WeatherDataLoader, merge_weather_with_prices
        weather_loader = WeatherDataLoader()
        weather_df = weather_loader.get_weather_features(df.index.min(), df.index.max())
        if weather_df is not None:
            df = merge_weather_with_prices(df, weather_df)
            logger.info(f"Weather features integrated. Total features: {len(df.columns)}")

    # Time-based Split (professor required 2024-10-21)
    split_date = pd.Timestamp("2024-10-21", tz='UTC')
    if split_date not in df.index:
        logger.warning(f"Fixed split date {split_date} not found in data. Falling back to 80/20 split.")
        split_idx = int(len(df) * 0.8)
        split_date = df.index[split_idx]
    else:
        split_idx = df.index.get_loc(split_date)
    
    logger.info(f"Split Index: {split_idx}, Split Date: {split_date}")
    
    if args.ablation:
        run_ablation(df, target_col)
        return

    # 3. Training & Prediction
    predictions = pd.DataFrame(index=df.index[split_idx:])
    
    # Naive
    logger.info("Running Naive Baselines...")
    from models.Naive.naive_baseline import NaiveBaseline
    naive = NaiveBaseline(df, target_col=target_col)
    for m in ['persistence', 'naive_24h', 'naive_7d']:
        predictions[f'Naive_{m}'] = naive.get_predictions(m).loc[predictions.index]

    baseline_rmse = None
    
    # XGBoost
    logger.info("Running XGBoost...")
    from models.XGBoost.xgboost_model import XGBoostModel
    xgb = XGBoostModel(df, target_col=target_col)
    xgb.preprocess()
    train_end_idx = sum(xgb.valid_indices < split_date)
    
    if args.tune_xgb:
        best_params, xgb_history = xgb.tune_hyperparameters(train_end_idx=train_end_idx, n_trials=args.n_trials)
        xgb.train(train_end_idx=train_end_idx, params=best_params)
    else:
        xgb.train(train_end_idx=train_end_idx)
    
    predictions['XGBoost'] = xgb.predict(start_date=split_date).reindex(predictions.index)

    # XGBoost: Lag + Cyclical Experiment (Automatically because requested)
    logger.info("Running Lag + Cyclical XGBoost Experiment...")
    # Find features that are either lags or cyclical (sin/cos)
    lag_cols = [c for c in xgb.feature_cols if 'lag' in c or 'sin' in c or 'cos' in c]
    lag_indices = [i for i, c in enumerate(xgb.feature_cols) if c in lag_cols]
    
    if lag_indices:
        import xgboost as xgb_lib
        X_train_lc = xgb.X_scaled[:train_end_idx, lag_indices]
        y_train_lc = xgb.y_scaled[:train_end_idx]
        
        xgb_lc = xgb_lib.XGBRegressor(objective='reg:squarederror', n_estimators=500, random_state=42)
        xgb_lc.fit(X_train_lc, y_train_lc.ravel())
        
        # Predict on test set
        idx = np.argmax(xgb.valid_indices >= split_date)
        X_test_lc = xgb.X_scaled[idx:, lag_indices]
        preds_lc_scaled = xgb_lc.predict(X_test_lc)
        preds_lc_real = xgb.scaler_y.inverse_transform(preds_lc_scaled.reshape(-1, 1)).ravel()
        predictions['XGB_LagCyclical'] = pd.Series(preds_lc_real, index=xgb.valid_indices[idx:]).reindex(predictions.index)
    else:
        logger.warning("No lag/cyclical features found for XGB_LagCyclical experiment.")

    # LSTM
    lstm_history = None
    lstm_optuna_history = None
    
    if args.tune_lstm and args.probabilistic:
        logger.info(f"Running LSTM Optuna Tuning ({args.n_trials} trials)...")
        from models.LSTM.lstm_optuna_tuner import LSTMOptunaOptimizer
        lstm_optimizer = LSTMOptunaOptimizer(df, target_col=target_col, split_idx=split_idx)
        best_lstm_params, lstm_optuna_history = lstm_optimizer.tune(n_trials=args.n_trials)
        
        # Train final model with best params
        logger.info("Training final LSTM with best Optuna params...")
        from models.LSTM.probabilistic_lstm import ProbabilisticLSTM
        lstm = ProbabilisticLSTM(
            df, target_col=target_col, 
            window_size=best_lstm_params.get('window_size', 72)
        )
        # Override hyperparams with Optuna results
        lstm.dropout = best_lstm_params.get('dropout', 0.2)
        lstm.batch_size = best_lstm_params.get('batch_size', 32)
        lstm.units = [
            best_lstm_params.get('units_1', 64),
            best_lstm_params.get('units_2', 128),
            best_lstm_params.get('units_3', 16)
        ]
        lstm_history = lstm.train(train_end_idx=split_idx)
        prob_results = lstm.predict(start_date=split_date)
        predictions['LSTM'] = prob_results['prediction'].values
        predictions['LSTM_uncertainty'] = prob_results['uncertainty'].values
        
    elif args.probabilistic:
        logger.info("Running Probabilistic LSTM (Monte Carlo Dropout)...")
        from models.LSTM.probabilistic_lstm import ProbabilisticLSTM
        lstm = ProbabilisticLSTM(df, target_col=target_col)
        lstm_history = lstm.train(train_end_idx=split_idx)
        prob_results = lstm.predict(start_date=split_date)
        predictions['LSTM'] = prob_results['prediction'].values
        predictions['LSTM_uncertainty'] = prob_results['uncertainty'].values
    else:
        logger.info("Running Standard LSTM...")
        from models.LSTM.lstm_model import LSTMModel
        lstm = LSTMModel(df.iloc[:split_idx], target_col=target_col)
        lstm.train()
        test_ctx = pd.concat([df.iloc[:split_idx].iloc[-lstm.window_size:], df.iloc[split_idx:]])
        y_pred, _ = lstm.predict_on_df(test_ctx)
        predictions['LSTM'] = y_pred[:len(predictions), -1]


    # 4. Evaluation
    predictions['Actual'] = df[target_col].iloc[split_idx:]
    
    # Get baseline RMSE first
    baseline_rmse, _, _, _, _ = evaluate_preds(predictions['Actual'], predictions['Naive_persistence'], 'Naive_persistence')
    
    metrics = []
    # Evaluate all prediction columns
    eval_cols = [c for c in predictions.columns if c not in ['Actual', 'LSTM_uncertainty', 'P05', 'P95']]
    for col in eval_cols:
        rmse, mae, n_pts, d_rmse, d_pct = evaluate_preds(predictions['Actual'], predictions[col], col, baseline_rmse)
        metrics.append({
            'Model': col, 'RMSE': rmse, 'MAE': mae, 'N_points': n_pts,
            'Delta_RMSE_vs_Naive': d_rmse, 'Delta_Pct_vs_Naive': d_pct
        })
        logger.info(f"{col}: RMSE={rmse:.4f}, MAE={mae:.4f} (N={n_pts})")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(LATEST_DIR / "metrics.csv", index=False)
    metrics_df.to_json(LATEST_DIR / "metrics.json", orient='records', indent=4)
    
    # Experimental Results: Model_History/ModelName/N.json
    for index, row in metrics_df.iterrows():
        m_name = str(row['Model']).replace(" ", "_")
        # Format requested by user
        trial_data = {
            "rmse": row['RMSE'],
            "mae": row['MAE'],
            "n_points": row['N_points'],
            "delta_naive": row['Delta_RMSE_vs_Naive'],
            "timestamp": datetime.now().isoformat(),
            "run_id": RUN_ID
        }
        # Add model-specific params if available
        if m_name == "XGBoost":
            trial_data.update(xgb.model.get_params())
        
        save_model_result(PROJECT_ROOT / "Analysis", m_name, trial_data)
    
    # === COMPREHENSIVE TRAINING HISTORY ===
    training_history = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    # LSTM History
    if lstm_history:
        training_history["models"]["LSTM"] = lstm_history
        # Add final metrics
        lstm_row = metrics_df[metrics_df['Model'] == 'LSTM'].iloc[0] if 'LSTM' in metrics_df['Model'].values else None
        if lstm_row is not None:
            training_history["models"]["LSTM"]["final_rmse"] = float(lstm_row['RMSE'])
            training_history["models"]["LSTM"]["final_mae"] = float(lstm_row['MAE'])
    
    # Add LSTM Optuna history if tuned
    if lstm_optuna_history:
        training_history["models"]["LSTM"]["optuna_history"] = lstm_optuna_history

    
    # XGBoost History
    training_history["models"]["XGBoost"] = {
        "hyperparameters": xgb.model.get_params(),
        "tuned": args.tune_xgb
    }
    if args.tune_xgb and 'xgb_history' in locals():
        training_history["models"]["XGBoost"]["optuna_history"] = xgb_history
    xgb_row = metrics_df[metrics_df['Model'] == 'XGBoost'].iloc[0] if 'XGBoost' in metrics_df['Model'].values else None
    if xgb_row is not None:
        training_history["models"]["XGBoost"]["final_rmse"] = float(xgb_row['RMSE'])
        training_history["models"]["XGBoost"]["final_mae"] = float(xgb_row['MAE'])
    
    # Save comprehensive training history
    # Clean NaN values from XGBoost params (NaN is not valid JSON)
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items() if not (isinstance(v, float) and np.isnan(v))}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    training_history = clean_nan(training_history)
    
    with open(LATEST_DIR / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=4, default=str)
    logger.info("Saved comprehensive training_history.json")

            
    predictions.to_csv(LATEST_DIR / "forecast_test.csv")

    # 5. Future 24h Forecast
    logger.info("Generating Future 24h Forecast...")
    future_preds = pd.DataFrame()
    
    # Naive Future (using last known values)
    future_preds['Naive_persistence'] = pd.Series([df[target_col].iloc[-1]]*24, index=pd.date_range(df.index[-1]+pd.Timedelta(hours=1), periods=24, freq='h'))
    
    # XGBoost Future
    if 'XGBoost' in predictions.columns:
        logger.info("XGBoost 24h forecast...")
        future_preds['XGBoost'] = xgb.predict_next_24h(df)
        
    # LSTM Future
    if 'LSTM' in predictions.columns:
        logger.info("LSTM 24h forecast...")
        f_lstm = lstm.predict_next_24h(df)
        future_preds['LSTM'] = f_lstm['prediction'].values
        if 'uncertainty' in f_lstm.columns:
            future_preds['LSTM_uncertainty'] = f_lstm['uncertainty'].values
            future_preds['P05'] = f_lstm['P05'].values
            future_preds['P95'] = f_lstm['P95'].values

    future_preds.to_csv(LATEST_DIR / "forecast_next24.csv")

    # 6. Visualizations (Organized Structure)
    logger.info("Generating plots (300 DPI)...")
    from utils.enhanced_visualizations import (
        create_comprehensive_comparison, 
        create_forecasting_insights, 
        create_feature_impact_analysis
    )
    
    # Forecasting Plots
    create_comprehensive_comparison(predictions, metrics_df, output_dir=LATEST_DIR / "Plots" / "Forecasting")
    
    # Error Analysis Plots
    create_forecasting_insights(predictions, output_dir=LATEST_DIR / "Plots" / "Error_Analysis")
    
    # Feature Plots
    if 'XGBoost' in predictions.columns:
        create_feature_impact_analysis(xgb.model, xgb.feature_cols, output_dir=LATEST_DIR / "Plots" / "Features")
    
    # Interactive Plots (Plotly)
    logger.info("Generating interactive HTML plots...")
    from utils.interactive_visualizations import create_interactive_forecast, create_interactive_error_analysis
    create_interactive_forecast(predictions, LATEST_DIR / "Plots" / "Interactive")
    create_interactive_error_analysis(predictions, LATEST_DIR / "Plots" / "Interactive")

    # Optuna Visuals
    optuna_hist = None
    if args.tune_xgb and 'xgb_history' in locals():
        logger.info("Generating Optuna Visuals and History...")
        optuna_hist = xgb_history
        optuna_dir = LATEST_DIR / "Optuna" / "XGBoost"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = optuna_dir / "optuna_history.json"
        with open(history_path, 'w') as f:
            json.dump(optuna_hist, f, indent=4)
        
        from utils.optuna_visualizer import visualize_optuna_history
        visualize_optuna_history(history_path, optuna_dir / "Plots")
        
        # Save detailed tuning log for Model_History
        tuning_log = []
        for t in optuna_hist['all_trials']:
            tuning_log.append({
                "objective": t['rmse'],
                "params": t['params'],
                "trial": t['trial_number'],
                "uid": f"xgb_tune_{RUN_ID}_{t['trial_number']}"
            })
        save_model_result(PROJECT_ROOT / "Analysis", "XGBoost_Tuning", tuning_log)

    # LSTM Optuna Visuals (same structure as XGBoost)
    if args.tune_lstm and lstm_optuna_history:
        logger.info("Generating LSTM Optuna Visuals and History...")
        lstm_optuna_dir = LATEST_DIR / "Optuna" / "LSTM"
        lstm_optuna_dir.mkdir(parents=True, exist_ok=True)
        
        lstm_history_path = lstm_optuna_dir / "optuna_history.json"
        with open(lstm_history_path, 'w') as f:
            json.dump(lstm_optuna_history, f, indent=4, default=str)
        
        from utils.optuna_visualizer import visualize_optuna_history
        visualize_optuna_history(lstm_history_path, lstm_optuna_dir / "Plots")
        
        # Save detailed tuning log
        lstm_tuning_log = []
        for t in lstm_optuna_history['all_trials']:
            lstm_tuning_log.append({
                "objective": t['rmse'],
                "params": t['params'],
                "trial": t['trial_number'],
                "uid": f"lstm_tune_{RUN_ID}_{t['trial_number']}"
            })
        save_model_result(PROJECT_ROOT / "Analysis", "LSTM_Tuning", lstm_tuning_log)

    # 7. Automated Report
    logger.info("Generating Automated Report...")
    from utils.report_generator import ReportGenerator
    reporter = ReportGenerator(RUN_ID, LATEST_DIR)
    reporter.generate(metrics_df, optuna_history=optuna_hist)
    
    logger.info(f"Run {RUN_ID} complete.")
    logger.info("Pipeline Execution Finished Successfully.")

if __name__ == "__main__":
    main()
