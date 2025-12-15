import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostModel:
    def __init__(self, df, target_col='GBP/mWh', feature_cols=None):
        """
        Day-ahead (t+24) XGBoost regressor.

        Args:
            df (pd.DataFrame): DataFrame with a DatetimeIndex.
            target_col (str): Target price column.
            feature_cols (list): Optional list of feature columns.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [c for c in df.columns if c != target_col]

        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        self.X_scaled = None
        self.y_scaled = None

    def preprocess(self):
        """
        Creates lag features and a day-ahead target y(t)=price(t+24),
        then scales X and y. Drops NaNs created by shifting (no ffill leakage).
        """
        # Add lags
        for lag in [24, 48, 168]:
            col_name = f"{self.target_col}_lag_{lag}"
            if col_name not in self.df.columns:
                self.df[col_name] = self.df[self.target_col].shift(lag)
            if col_name not in self.feature_cols:
                self.feature_cols.append(col_name)

        # Day-ahead target
        self.df["target_t_plus_24"] = self.df[self.target_col].shift(-24)

        # Drop NaNs from shifting (IMPORTANT)
        needed = list(self.feature_cols) + ["target_t_plus_24"]
        self.df = self.df.dropna(subset=needed).copy()

        X = self.df[self.feature_cols].values
        y = self.df[["target_t_plus_24"]].values

        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)

        return self.X_scaled, self.y_scaled

    def train(self, train_end_idx, params=None, verbose=True, validation_split=0.1):
        """
        Trains the XGBoost model using the preprocessed arrays.
        
        Args:
            train_end_idx (int): index split on self.X_scaled/y_scaled
            params (dict): XGBoost params
            verbose (bool): If True, prints training progress.
            validation_split (float): Fraction of training data to use for validation.
        """
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1200,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1.0,
                'n_jobs': -1,
                'random_state': 42
            }

        # Full training set
        X_full = self.X_scaled[:train_end_idx]
        y_full = self.y_scaled[:train_end_idx]
        
        # Create validation split (time-series aware: take last chunk)
        split_point = int(len(X_full) * (1 - validation_split))
        
        X_train = X_full[:split_point]
        y_train = y_full[:split_point]
        X_val = X_full[split_point:]
        y_val = y_full[split_point:]
        
        self.model = xgb.XGBRegressor(**params)
        
        print("\n--- Starting XGBoost Training ---")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=verbose
        )
        print("--- XGBoost training completed ---\n")

    def tune_hyperparameters(self, train_end_idx, n_iter=10):
        """
        Performs Optuna optimization to find better hyperparameters.
        Returns the best parameters found.
        """
        print(f"Starting XGBoost Hyperparameter Tuning with Optuna (n_trials={n_iter})...")
        
        X_train_full = self.X_scaled[:train_end_idx]
        y_train_full = self.y_scaled[:train_end_idx]
        
        # Define TimeSeriesSplit for stable cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        def objective(trial):
            # Hyperparameter Search Space
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'n_jobs': -1,
                'random_state': 42
            }
            
            # Cross-Validation Loop
            rmse_scores = []
            
            for train_index, val_index in tscv.split(X_train_full):
                X_t, X_v = X_train_full[train_index], X_train_full[val_index]
                y_t, y_v = y_train_full[train_index], y_train_full[val_index]
                
                model = xgb.XGBRegressor(**params)
                
                # Pruning callback is tricky with XGBoost scikit-learn API + Optuna integration
                # For simplicity in this script, we just fit and evaluate.
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=20, verbose=False)
                
                preds_scaled = model.predict(X_v)
                # Note: We evaluate on scaled data for speed/stability during tuning, 
                # or unscaled if we want real RMSE. 
                # RandomizedSearchCV used 'neg_root_mean_squared_error' which usually runs on y passed to it.
                # If y passed is scaled, then RMSE is scaled.
                # To be consistent with previous approach, we stick to minimized scaled RMSE.
                
                rmse = np.sqrt(mean_squared_error(y_v, preds_scaled))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_iter)
        
        print(f"Best Score (Scaled RMSE): {study.best_value:.4f}")
        print(f"Best Params: {study.best_params}")
        
        return study.best_params

    def predict(self, test_start_idx):
        """
        Predicts day-ahead prices for samples >= test_start_idx.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet.")

        X_test = self.X_scaled[test_start_idx:]
        if len(X_test) == 0:
            return np.array([])

        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred.flatten()

    def evaluate(self, test_start_idx):
        """
        Evaluates RMSE/MAE on day-ahead target_t_plus_24.
        """
        y_pred = self.predict(test_start_idx)
        y_true = self.df["target_t_plus_24"].values[test_start_idx:]

        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {'rmse': rmse, 'mae': mae, 'y_pred': y_pred, 'y_true': y_true}
