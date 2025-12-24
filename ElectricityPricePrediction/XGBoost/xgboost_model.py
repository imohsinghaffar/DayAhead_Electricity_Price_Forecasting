import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostModel:
    """XGBoost regressor configured for day-ahead (t+24) forecasting.

    Creates internal day-ahead target y_day_ahead = price.shift(-24), adds lag features,
    scales data for training, and provides Optuna tuning returning real-unit RMSE.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "Price", feature_cols=None):
        self.df = df.copy()
        self.target_col = target_col
        # Defaults if not provided, excluding target from features
        self.feature_cols = feature_cols if feature_cols is not None else [c for c in df.columns if c != target_col]

        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X_scaled = None
        self.y_scaled = None

    def preprocess(self):
        # 1. Create Lag Features
        # Conditional add based on data length to avoid empty set on small data
        n_rows = len(self.df)
        for lag in [24, 48, 168]:
            if n_rows <= lag:
                print(f"[XGBoost] Warning: Dataset length ({n_rows}) too small for lag {lag}. Skipping.")
                continue
                
            col = f"{self.target_col}_lag_{lag}"
            if col not in self.df.columns:
                self.df[col] = self.df[self.target_col].shift(lag)
            if col not in self.feature_cols:
                self.feature_cols.append(col)

        # 2. Function day-ahead target: y(t) = Price(t+24)
        # Shift -24: Row T gets value from T+24.
        self.df["target_t_plus_24"] = self.df[self.target_col].shift(-24)

        # 3. Drop NaNs
        needed = list(self.feature_cols) + ["target_t_plus_24"]
        self.df = self.df.dropna(subset=needed).copy()
        
        # Store valid indices for alignment
        self.valid_indices = self.df.index

        # 4. Scale
        X = self.df[self.feature_cols].values
        y = self.df[["target_t_plus_24"]].values

        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)

        return self.X_scaled, self.y_scaled

    def train(self, train_end_idx: int, params: dict | None = None, verbose: bool = True, validation_split: float = 0.1):
        # NOTE: train_end_idx is interpreted as integer index in the PREPROCESSED (Valid) array
        if params is None:
            # Default Baseline Params
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'n_jobs': -1,
                'random_state': 42
            }

        X_full = self.X_scaled[:train_end_idx]
        y_full = self.y_scaled[:train_end_idx]

        # Time-series aware validation split (last 10% of train set)
        split_point = int(len(X_full) * (1 - validation_split))
        X_train, X_val = X_full[:split_point], X_full[split_point:]
        y_train, y_val = y_full[:split_point], y_full[split_point:]

        self.model = xgb.XGBRegressor(**params)
        
        if verbose:
            print(f"\n[XGBoost] Training with {len(X_train)} samples, Validating on {len(X_val)} samples.")

        self.model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_val, y_val.ravel())],
            early_stopping_rounds=50,
            verbose=verbose
        )
        
        if verbose:
            print("[XGBoost] Training completed.\n")

    def tune_hyperparameters(self, train_end_idx: int, n_trials: int = 20):
        # Same as before
        X_tune = self.X_scaled[:train_end_idx]
        y_tune = self.y_scaled[:train_end_idx]
        y_tune_real = self.scaler_y.inverse_transform(y_tune).ravel()
        
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'random_state': 42
            }

            scores = []
            for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_tune)):
                X_t, X_v = X_tune[train_idx], X_tune[val_idx]
                y_t_s, y_v_s = y_tune[train_idx].ravel(), y_tune[val_idx].ravel()
                y_v_real = y_tune_real[val_idx] # Real target

                model = xgb.XGBRegressor(**params)
                model.fit(X_t, y_t_s, verbose=False)

                preds_s = model.predict(X_v)
                preds_real = self.scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel()

                rmse = np.sqrt(mean_squared_error(y_v_real, preds_real))
                scores.append(rmse)

            return np.mean(scores)

        print(f"\n[XGBoost] Starting Optuna Tuning ({n_trials} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def predict(self, start_date):
        """
        Predicts from `start_date` onwards. Returns pd.Series aligned with index.
        """
        if self.model is None:
            raise RuntimeError("Model NOT trained.")
            
        # Find integer index corresponding to start_date in VALID indices
        # We want >= start_date
        mask = self.valid_indices >= start_date
        if not mask.any():
            return pd.Series(dtype=float)
            
        test_start_int_idx = np.argmax(mask) # First True
        
        X_test = self.X_scaled[test_start_int_idx:]
        indices = self.valid_indices[test_start_int_idx:]
        
        if len(X_test) == 0:
            return pd.Series(dtype=float)
            
        preds_scaled = self.model.predict(X_test)
        preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        
        return pd.Series(preds_real, index=indices)
