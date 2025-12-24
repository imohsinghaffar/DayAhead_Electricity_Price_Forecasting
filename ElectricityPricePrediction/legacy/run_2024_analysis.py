import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce noise

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader_2024 import DataLoader2024
from Naive.naive_baselines import NaiveBaseline
from XGBoost.xgboost_model import XGBoostModel

def evaluate_robust(y_true, y_pred, model_name):
    common_idx = y_true.index.intersection(y_pred.index)
    if len(common_idx) == 0:
        return {'rmse': np.nan, 'mae': np.nan}
    
    y_t = y_true.loc[common_idx]
    y_p = y_pred.loc[common_idx]
    
    # Drop NaNs (e.g. at the start of forecast)
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]
    
    if len(y_t) == 0:
        return {'rmse': np.nan, 'mae': np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    
    print(f"[{model_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {'rmse': rmse, 'mae': mae}

def main():
    parser = argparse.ArgumentParser()
    # User provided 2024 data. Let's test on the last 2 months (Nov-Dec)
    parser.add_argument('--test_start', type=str, default='2024-11-01')
    parser.add_argument('--test_end', type=str, default=None, help='Optional end date for test set')
    parser.add_argument('--skip_lstm', action='store_true', help='Skip LSTM model training')
    args = parser.parse_args()
    
    # 1. Load Data
    print("\n=== Step 1: Loading 2024 Data ===")
    loader = DataLoader2024()
    df = loader.load_data()
    
    test_start_dt = pd.to_datetime(args.test_start)
    train_df = df[df.index < test_start_dt].copy()
    
    if args.test_end:
        test_end_dt = pd.to_datetime(args.test_end)
        test_df = df[(df.index >= test_start_dt) & (df.index < test_end_dt)].copy()
    else:
        test_df = df[df.index >= test_start_dt].copy()
    print(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")
    
    preds_df = pd.DataFrame(index=test_df.index)
    preds_df['Actual'] = test_df['Price']
    metrics = {}
    
    # 2. Naive Baselines
    print("\n=== Step 2: Naive Baselines ===")
    naive = NaiveBaseline(df, target_col='Price') # Use full df to allow lookback
    
    preds_df['Naive_24h'] = naive.predict_24h().reindex(test_df.index)
    metrics['Naive_24h'] = evaluate_robust(test_df['Price'], preds_df['Naive_24h'], "Naive_24h")
    
    preds_df['Naive_7d'] = naive.predict_7d().reindex(test_df.index)
    metrics['Naive_7d'] = evaluate_robust(test_df['Price'], preds_df['Naive_7d'], "Naive_7d")
    
    # 3. XGBoost
    print("\n=== Step 3: XGBoost ===")
    xgb_model = XGBoostModel(df, target_col='Price')
    xgb_model.preprocess()
    
    cutoff_idx = len(xgb_model.df[xgb_model.df.index < test_start_dt])
    
    # Optional: Quick tune or use standard? 
    # Use standard to save time, or quick tune.
    # Let's use the 'tune_hyperparameters' but with few iters for speed, 
    # OR just use the previous best params which were quite good.
    # Given the new features (Coal/Gas), retuning is better.
    # But for script stability, I'll use the method I added.
    try:
        # Reduce n_iter to 2 for speed and less noise, user cares about final training visibility
        print("Quick Tuning XGBoost...")
        best_params = xgb_model.tune_hyperparameters(train_end_idx=cutoff_idx, n_iter=2) 
        print(f"Best Params Found: {best_params}")
        # Add explicit print for user visibility
        print("\n" + "="*40)
        print(" STARTING FINAL XGBOOST TRAINING ")
        print("="*40 + "\n")
        xgb_model.train(train_end_idx=cutoff_idx, params=best_params, verbose=True)
    except Exception as e:
        print(f"XGB tuning failed ({e}), using default training.")
        xgb_model.train(train_end_idx=cutoff_idx, verbose=True)
        
    y_pred_xgb = xgb_model.predict(test_start_idx=cutoff_idx)
    
    # Align XGB
    # XGB predict returns 'target_t_plus_24'.
    # timesteps: test_start_dt -> end.
    # The first prediction at cutoff_idx is for cutoff_idx time? No.
    # X[cutoff] corresponds to dfrow[cutoff]. target is shift(-24).
    # So X[t] predicts Price[t+24].
    # So if X[t].index is T, prediction is for T+24h.
    
    xgb_timestamps = xgb_model.df.index[cutoff_idx:] + pd.Timedelta(hours=24)
    xgb_series = pd.Series(y_pred_xgb, index=xgb_timestamps)
    preds_df['XGBoost'] = xgb_series.reindex(test_df.index)
    metrics['XGBoost'] = evaluate_robust(test_df['Price'], preds_df['XGBoost'], "XGBoost")
    
    # 4. LSTM (Residual Learning)
    if not args.skip_lstm:
        print("\n=== Step 4: LSTM (Residual Learning) ===")
        
        # Lazy import to avoid segfaults if not used or environment issues
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            import tensorflow as tf
            tf.random.set_seed(42)
            from LSTM.lstm_model import LSTMModel
            
            # === ACTIVE OPTUNA TUNING ===
            print("\n" + "="*40)
            print(" ACTIVE OPTUNA TUNING (LSTM) ")
            print("="*40)
            
            # Prepare data for tuning
            full_naive_24h = naive.predict_24h()
            train_df_with_naive = train_df.copy()
            train_df_with_naive['Naive_24h'] = full_naive_24h.reindex(train_df.index)
            train_df_with_naive['Residual'] = train_df_with_naive['Price'] - train_df_with_naive['Naive_24h']
            
            def objective(trial):
                # Hyperparameters to tune
                u1 = trial.suggest_int('units_1', 32, 128)
                u2 = trial.suggest_int('units_2', 16, 64)
                do = trial.suggest_float('dropout', 0.1, 0.5)
                lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                bs = trial.suggest_categorical('batch_size', [32, 64, 128])
                
                # Split train further into train/val for tuning OR use internal split
                # Here we use internal split of the 'train_df' we have.
                # Simplest way: instantiate model with train_df and let it split
                model = LSTMModel(train_df_with_naive, target_col='Residual', window_size=168, forecast_horizon=24)
                model.train(
                    epochs=10, # Short epochs for tuning
                    batch_size=bs,
                    units_1=u1,
                    units_2=u2,
                    dropout=do,
                    learning_rate=lr,
                    train_ratio=0.8
                )
                
                # Predict on last portion (approx val)
                # To be robust, we just want to minimize 'val_loss' which is available in History 
                # but our class doesn't return history easily.
                # So we simply return the last val_loss from internal training? 
                # Currently .train() doesn't return anything.
                # We need to run predict on a held-out set.
                
                # Let's trust the 'train_df' split.
                # Actually, running full training here on every trial might be slow.
                # But it's what the user asked for.
                
                # Mock return for integration speed if needed, but let's try to be real.
                # Ideally, we return the internal validation metric.
                # For now, let's assume we can't easily get it without refactoring.
                # So we return 0.0 (Dummy) to proceed? NO, that breaks optimization.
                # Let's just use a fixed set of good params if we can't optimize easily?
                # NO, user asked for "working along".
                
                # QUICK FIX: Modify LSTM class to return history? 
                # Or just use the global hardcoded ones as fallback if this is too complex.
                # But I see 'metrics' option in compile.
                
                # Hack: We use a separate small validation set here manually.
                split_idx = int(len(train_df_with_naive) * 0.8)
                t_df = train_df_with_naive.iloc[:split_idx]
                v_df = train_df_with_naive.iloc[split_idx:]
                
                m = LSTMModel(t_df, target_col='Residual', window_size=168, forecast_horizon=24)
                m.train(epochs=8, batch_size=bs, units_1=u1, units_2=u2, dropout=do, learning_rate=lr)
                
                # Predict on v_df
                # We need context from t_df
                combined = pd.concat([t_df.iloc[-200:], v_df])
                preds, _ = m.predict_on_df(combined)
                
                # Align to v_df
                # Similar logic to main loop
                # Just take the RMSE of the residuals directly
                # v_df target is 'Residual'
                
                # Align length
                if len(preds) > len(v_df):
                    p = preds[-len(v_df):, -1] # Last step
                    t = v_df['Residual'].values
                    # Check NaNs
                    mask = ~np.isnan(p) & ~np.isnan(t)
                    return np.sqrt(mean_squared_error(t[mask], p[mask]))
                return 100.0 # Fail

            print("Running Quick Optuna Study (5 Trials)...")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=5)
            
            print(f"Best Optuna Params: {study.best_params}")
            lstm_params = study.best_params
            lstm_params['epochs'] = 30 # Train longer for final model
            
            
            # Check if we can find a 'best_params.txt' or similar? 
            # For now, I will assume the user (me) will update this script with best params 
            # OR I'll update it in the next turn after Optuna finishes.
            
            # Prepare data for residual learning
            # We need the Naive_24h prediction for the entire df to calculate residuals
            # full_naive_24h = naive.predict_24h() (Calculated above)
            
            # Calculate residuals for training
            train_df_with_naive = train_df.copy()
            train_df_with_naive['Naive_24h'] = full_naive_24h.reindex(train_df.index)
            train_df_with_naive['Residual'] = train_df_with_naive['Price'] - train_df_with_naive['Naive_24h']
            
            # Prepare test_df for reconstruction
            test_df_with_naive = test_df.copy()
            test_df_with_naive['Naive_24h'] = full_naive_24h.reindex(test_df.index)
            
            print(f"Training LSTM with params: {lstm_params}")
            # Train on Residuals
            # The LSTM model expects 'target_col' to be present in the df passed to it.
            # We also need to ensure the window_size and forecast_horizon are consistent.
            lstm_residual_model = LSTMModel(train_df_with_naive, target_col='Residual', window_size=168, forecast_horizon=24) 
            lstm_residual_model.train(
                epochs=lstm_params['epochs'],
                batch_size=lstm_params['batch_size'],
                units_1=lstm_params['units_1'],
                units_2=lstm_params['units_2'],
                dropout=lstm_params['dropout'],
                learning_rate=lstm_params['learning_rate']
            )
            
            # Predict Residuals on Test Set
            # We need to construct the sliding window from train+test to avoid gap at start
            # We use -200 to ensure we have enough history (168 window + 24 horizon) to predict the very first test point
            combined_df = pd.concat([train_df_with_naive.iloc[-200:], test_df_with_naive])
            
            # Predict using the CORRECT model variable
            resid_vector_pred, _ = lstm_residual_model.predict_on_df(combined_df)
            
            # Extract 24th step prediction (t+24 hours ahead)
            final_resid = resid_vector_pred[:, -1]
            
            # ALIGNMENT:
            # lstm_model._make_xy uses indices.
            # The loop runs from 'window_size' to 'n - horizon + 1'.
            # For a given i, the prediction vector corresponds to combined_df.iloc[i : i+horizon].
            # The last element (index -1) corresponds to combined_df.iloc[i + horizon - 1].
            
            start_idx = lstm_residual_model.window_size + lstm_residual_model.forecast_horizon - 1
            pred_times = combined_df.index[start_idx:]
            
            # Ensure lengths match (sometimes slice can be off by 1 depending on implementation nuances)
            # The loop length is len(final_resid).
            # pred_times = pred_times[:len(final_resid)] # Safety clip
            
            # Reconstruct Price: Pred = Naive (at pred_time) + Resid (at pred_time)
            # We need Naive_24h values aligned to pred_times
            naive_at_pred = combined_df.loc[pred_times, 'Naive_24h']
            
            lstm_forecast_series = pd.Series(
                naive_at_pred.values + final_resid, 
                index=pred_times
            )
            
            # Save to preds_df (align to test_df)
            preds_df['LSTM'] = lstm_forecast_series.reindex(test_df.index)
            
            # Calculate Metrics on common index (drop NaNs from start/end mismatches)
            valid_preds = preds_df['LSTM'].dropna()
            valid_actuals = test_df.loc[valid_preds.index, 'Price']
            
            rmse_lstm = np.sqrt(mean_squared_error(valid_actuals, valid_preds))
            mae_lstm = mean_absolute_error(valid_actuals, valid_preds)
            print(f"[LSTM - Residual] RMSE: {rmse_lstm:.4f} | MAE: {mae_lstm:.4f}")
            
            metrics['LSTM'] = {'rmse': rmse_lstm, 'mae': mae_lstm} # Manually add or use evaluate_robust if possible
            # But for Residual we calculated manually above. So just store it.
            # evaluate_robust prints again, maybe redundant but harmless?
            # Actually evaluate_robust calculates on its own. 
            # Let's just store the manually calculated ones to match the print.
            metrics['LSTM'] = {'rmse': rmse_lstm, 'mae': mae_lstm}
            
        except ImportError:
            print("TensorFlow not installed or crashed. Skipping LSTM.")
            metrics['LSTM'] = {'rmse': np.nan, 'mae': np.nan}
            preds_df['LSTM'] = np.nan
            # return # DON'T RETURN, CONTINUE
    
    else:
        print("\n=== Step 4: Skipping LSTM (User Request) ===")
        metrics['LSTM'] = {'rmse': np.nan, 'mae': np.nan}
        preds_df['LSTM'] = np.nan
    
    # === Step 5: Ensemble (Average) ===
    print("\n=== Step 5: Ensemble Model (XGB + LSTM) ===")
    preds_df['Ensemble'] = (preds_df['XGBoost'] + preds_df['LSTM']) / 2
    metrics['Ensemble'] = evaluate_robust(test_df['Price'], preds_df['Ensemble'], "Ensemble")
    
    # === Step 6: Visualization & Reporting ===
    print("\n=== Step 6: Generating Visualizations (Analysis/plots/) ===")
    os.makedirs('Analysis/plots', exist_ok=True)
    
    # 1. Forecast Comparison (Full)
    plt.figure(figsize=(15, 7))
    plt.plot(preds_df.index, preds_df['Actual'], 'k-', label='Actual', lw=1.5, alpha=0.9)
    plt.plot(preds_df.index, preds_df['Naive_24h'], '--', label='Naive 24h', alpha=0.5)
    plt.plot(preds_df.index, preds_df['XGBoost'], label='XGBoost', alpha=0.7)
    plt.plot(preds_df.index, preds_df['LSTM'], label='LSTM', alpha=0.7)
    plt.plot(preds_df.index, preds_df['Ensemble'], 'r-', label='Ensemble', lw=1.5)
    plt.legend()
    plt.title('2024 Electricity Price Forecast: Model Comparison')
    plt.ylabel('Price (EUR/MWh)')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('Analysis/plots/1_forecast_comparison.png')
    plt.close()
    
    # 2. Zoom-in (Volatility) - First week of Dec
    zoom_start = '2024-12-01'
    zoom_end = '2024-12-08'
    subset = preds_df[(preds_df.index >= zoom_start) & (preds_df.index < zoom_end)]
    
    if len(subset) > 0:
        plt.figure(figsize=(15, 7))
        plt.plot(subset.index, subset['Actual'], 'k-o', label='Actual', lw=2)
        plt.plot(subset.index, subset['Naive_24h'], '--', label='Naive 24h', alpha=0.7)
        plt.plot(subset.index, subset['Ensemble'], 'r-o', label='Ensemble', lw=2)
        plt.legend()
        plt.title(f'Volatility Zoom-in ({zoom_start} to {zoom_end})')
        plt.ylabel('Price (EUR/MWh)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Analysis/plots/2_forecast_zoom.png')
        plt.close()
    
    # 3. Metrics Comparison (Bar Chart)
    metrics_df = pd.DataFrame(metrics).T
    plt.figure(figsize=(12, 7)) # Slightly wider
    x = np.arange(len(metrics_df))
    width = 0.35
    
    rmse_bars = plt.bar(x - width/2, metrics_df['rmse'], width, label='RMSE', color='indianred')
    mae_bars = plt.bar(x + width/2, metrics_df['mae'], width, label='MAE', color='steelblue')
    
    # Add labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    add_labels(rmse_bars)
    add_labels(mae_bars)
    
    plt.ylabel('Error (EUR/MWh)')
    plt.title('Model Error Metrics (Lower is Better)')
    plt.xticks(x, metrics_df.index)
    plt.legend()
    plt.grid( axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Analysis/plots/3_metrics_comparison.png')
    plt.close()
    
    # 4. Error Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    residuals = preds_df['Ensemble'] - preds_df['Actual']
    plt.hist(residuals.dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='k', linestyle='--', lw=1)
    plt.title('Ensemble Residuals Distribution (Prediction - Actual)')
    plt.xlabel('Error (EUR/MWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Analysis/plots/4_error_distribution.png')
    plt.close()
    
    # 5. Training History (Placeholder or from LSTM)
    # Since we don't return history object easily in this script structure without refactoring LSTM class deeply,
    # We will skip plotting it here or we would need to modify LSTM.train to return history.
    # For now, let's acknowledge we saved 4 plots.
    
    print("Graphs generated in Analysis/plots/:")
    print("1. 1_forecast_comparison.png")
    print("2. 2_forecast_zoom.png")
    print("3. 3_metrics_comparison.png")
    print("4. 4_error_distribution.png")
    
    # 6. Feature Importance (XGBoost) - Requested by User
    try:
        importance = xgb_model.model.feature_importances_
        feature_names = xgb_model.feature_cols
        
        # Sort features
        sorted_idx = np.argsort(importance)[-10:] # Top 10
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color='teal')
        plt.title('Top 10 Feature Importance (XGBoost)')
        plt.xlabel('Importance Score')
        plt.grid( axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('Analysis/plots/5_feature_importance.png')
        plt.close()
        print("5. 5_feature_importance.png")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

    # 7. Correlation Plot (Price vs Wind) - Impact Analysis
    if 'Wind_Speed' in df.columns:
        plt.figure(figsize=(10, 6))
        # Hexbin for density
        plt.hexbin(df['Wind_Speed'], df['Price'], gridsize=30, cmap='Blues')
        plt.colorbar(label='Count')
        plt.xlabel('Wind Speed (10m)')
        plt.ylabel('Electricity Price (EUR/MWh)')
        plt.title('Impact Analysis: Wind Speed vs Price')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Analysis/plots/6_correlation_wind_price.png')
        plt.close()
        print("6. 6_correlation_wind_price.png")

    with open("Analysis/analysis_results_final.txt", "w") as f:
        f.write("2024 Final Analysis Results\n===========================\n")
        metrics_df.to_string(f)
        print("\nFinal Metrics:")
        print(metrics_df)
    
    # 8. Run Next Day Forecast (Streamlined)
    print("\n" + "="*40)
    print(" STREAMLINING: Running Next Day Forecast ")
    print("="*40 + "\n")
    try:
        from Analysis.forecast_next_day import predict_next_day
        # Pass the optimized params if they exist, otherwise None
        if 'lstm_params' in locals():
            predict_next_day(lstm_params=lstm_params)
        else:
            predict_next_day()
    except Exception as e:
        print(f"Error running forecast step: {e}")

if __name__ == "__main__":
    main()
