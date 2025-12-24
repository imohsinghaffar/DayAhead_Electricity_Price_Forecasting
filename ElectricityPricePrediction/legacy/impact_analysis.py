import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import xgboost as xgb

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from XGBoost.xgboost_model import XGBoostModel

def load_data(filepath):
    """
    Loads the dataset properly handling time index.
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

def main():
    # 1. Load Data
    # Use the absolute path logic to find the data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data/re_fixed_multivariate_timeseires.csv')
    
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    if df is None:
        print("Could not load data. Exiting.")
        return

    # Check for Oil Price and other features
    print("Available features:", df.columns.tolist())
    
    target_col = 'price' if 'price' in df.columns else 'GBP/mWh'
    if target_col not in df.columns:
        # Fallback
        target_col = df.columns[0]
    
    print(f"Target Variable: {target_col}")

    # 2. Train XGBoost Model
    print("\nTraining XGBoost to determine feature importance...")
    
    # We use the whole dataset (or a large train split) to get robust feature importance
    # Let's straightforwardly split as in model_comparison
    # Train: < 2019, but actually for feature importance we might want to use as much data as possible.
    # We will stick to the standard train/test split to be consistent.
    test_start_date = '2019-01-01'
    df.sort_index(inplace=True)
    train_df = df[df.index < test_start_date].copy()
    
    if train_df.empty:
         # use 80% split if dates don't align
         train_size = int(len(df) * 0.8)
         train_df = df.iloc[:train_size].copy()
    
    # Initialize Model
    # We pass the full dataframe to initialization as per class design
    xgb_model = XGBoostModel(df, target_col=target_col)
    xgb_model.preprocess()
    
    # Train
    xgb_model.train(train_end_idx=len(train_df))
    
    # 3. Extract Feature Importances
    if xgb_model.model is None:
        print("Model failed to train.")
        return

    # Get feature importance
    # XGBoost provides different types: 'weight', 'gain', 'cover'. 'gain' is usually most interpretable for "impact".
    importance_type = 'gain'
    importances = xgb_model.model.get_booster().get_score(importance_type=importance_type)
    
    # The keys in `importances` are 'f0', 'f1' etc. corresponding to feature_cols
    # We need to map them back to names.
    # XGBoostModel.feature_cols holds the names.
    feature_names = xgb_model.feature_cols
    
    # Map f0 -> feature_name
    min_len = len(feature_names)
    # Note: feature_importances_ property often returns array aligned with X columns
    # Let's try the array property first which is safer if sklearn API is used
    if hasattr(xgb_model.model, 'feature_importances_'):
        vals = xgb_model.model.feature_importances_
        # vals corresponds to feature_names order
        importance_dict = dict(zip(feature_names, vals))
    else:
        # Fallback
        importance_dict = importances

    # Sort
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=False)
    
    # Separate names and values
    sorted_names = [x[0] for x in sorted_importance]
    sorted_vals = [x[1] for x in sorted_importance]

    print("\nFeature Importances (Top 10):")
    for name, val in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"{name}: {val:.4f}")

    # 4. Plot
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_names, sorted_vals, color='teal')
    plt.title(f'XGBoost Feature Importance (Feature Impact on {target_col})')
    plt.xlabel('Relative Importance (Gain)')
    plt.tight_layout()
    
    plot_dir = os.path.join(base_dir, 'Analysis', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'feature_importance.png')
    plt.savefig(save_path)
    print(f"\nFeature importance plot saved to: {save_path}")

if __name__ == "__main__":
    main()
