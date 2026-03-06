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

    def __init__(self, df: pd.DataFrame, target_col: str = "Price", feature_cols=None, use_residuals=True, confidence_scaling: float = 1.0):
        self.df = df.copy()
        self.target_col = target_col
        self.use_residuals = use_residuals
        self.confidence_scaling = confidence_scaling
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
            col = f"{self.target_col}_lag_{lag}"
            if col in self.df.columns:
                if col not in self.feature_cols:
                    self.feature_cols.append(col)
                continue
                
            if n_rows <= lag:
                print(f"[XGBoost] Warning: Dataset length ({n_rows}) too small for lag {lag}. Skipping.")
                continue
                
            self.df[col] = self.df[self.target_col].shift(lag)
            if col not in self.feature_cols:
                self.feature_cols.append(col)

        # 2. Function day-ahead target: y(t) = Price(t+24)
        # Shift -24: Row T gets value from T+24.
        # If use_residuals: target = Price(t+24) - Price(t)
        if self.use_residuals:
            self.df["target_t_plus_24"] = self.df[self.target_col].shift(-24) - self.df[self.target_col]
        else:
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
        self.X_train, self.X_val = X_full[:split_point], X_full[split_point:]
        self.y_train, self.y_val = y_full[:split_point], y_full[split_point:]

        self.model = xgb.XGBRegressor(**params)
        
        if verbose:
            print(f"\n[XGBoost] Training with {len(self.X_train)} samples, Validating on {len(self.X_val)} samples.")

        self.model.fit(
            self.X_train, self.y_train.ravel(),
            eval_set=[(self.X_val, self.y_val.ravel())],
            early_stopping_rounds=50,
            verbose=verbose
        )
        
        # Calculate Validation Residuals for Probabilistic Intervals
        # res = actual - predicted
        val_preds = self.model.predict(self.X_val)
        self.val_residuals = self.scaler_y.inverse_transform(self.y_val).ravel() - \
                             self.scaler_y.inverse_transform(val_preds.reshape(-1, 1)).ravel()
        
        self.p05_offset = np.percentile(self.val_residuals, 5)
        self.p95_offset = np.percentile(self.val_residuals, 95)
        
        if verbose:
            print(f"[XGBoost] Training complete. Val Residual P05: {self.p05_offset:.2f}, P95: {self.p95_offset:.2f}\n")

    def tune_hyperparameters(self, train_end_idx: int, n_trials: int = 20, baseline_params: dict | None = None):
        """Runs Optuna optimization and returns trial history for reporting."""
        X_tune = self.X_scaled[:train_end_idx]
        y_tune = self.y_scaled[:train_end_idx]
        y_tune_real = self.scaler_y.inverse_transform(y_tune).ravel()
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)

        # 1. Calculate Baseline Performance (if provided or using defaults)
        if baseline_params is None:
            baseline_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        
        def run_eval(params):
            scores = []
            for train_idx, val_idx in tscv.split(X_tune):
                model = xgb.XGBRegressor(**params)
                model.fit(X_tune[train_idx], y_tune[train_idx].ravel(), verbose=False)
                preds = model.predict(X_tune[val_idx])
                preds_real = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
                scores.append(np.sqrt(mean_squared_error(y_tune_real[val_idx], preds_real)))
            return np.mean(scores)

        baseline_rmse = run_eval(baseline_params)

        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
            return run_eval(params)

        import optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_rmse = study.best_value
        improvement_rmse = baseline_rmse - best_rmse
        improvement_pct = (improvement_rmse / baseline_rmse * 100) if baseline_rmse != 0 else 0

        # Structured History for Report
        history = {
            "study_metadata": {
                "n_trials": n_trials,
                "best_rmse": best_rmse,
                "best_params": best_params,
                "baseline_rmse": baseline_rmse,
                "baseline_params": baseline_params,
                "improvement": {
                    "rmse_delta": improvement_rmse,
                    "rmse_pct": improvement_pct
                }
            },
            "all_trials": [
                {"trial_number": t.number, "rmse": t.value, "params": t.params, "state": t.state.name}
                for t in study.trials
            ]
        }
        
        return best_params, history

    def predict(self, start_date, probabilistic=False):
        """Sequence prediction for test set. Returns DataFrame if probabilistic."""
        if self.model is None: raise RuntimeError("Model NOT trained.")
        mask = self.valid_indices >= start_date
        if not mask.any(): return pd.Series(dtype=float)
        idx = np.argmax(mask)
        X_test = self.X_scaled[idx:]
        indices = self.valid_indices[idx:]
        
        # Point Prediction
        preds_raw = self.scaler_y.inverse_transform(self.model.predict(X_test).reshape(-1, 1)).ravel()
        
        if self.use_residuals:
            base_prices = self.df.loc[indices, self.target_col].values
            preds_final = preds_raw + base_prices
        else:
            preds_final = preds_raw

        if not probabilistic:
            return pd.Series(preds_final, index=indices)
            
        # Probabilistic: use pre-calculated offsets from validation set
        res = pd.DataFrame(index=indices)
        res['prediction'] = preds_final
        res['P05'] = preds_final + (self.p05_offset * self.confidence_scaling)
        res['P95'] = preds_final + (self.p95_offset * self.confidence_scaling)
        res['uncertainty'] = (res['P95'] - res['P05']) / 4
        
        return res

    def predict_next_24h(self, latest_df: pd.DataFrame, probabilistic=False):
        """
        Builds future features and predicts next 24 hours.
        Returns DataFrame if probabilistic.
        """
        data = latest_df.copy()
        for lag in [24, 48, 168]:
            col = f"{self.target_col}_lag_{lag}"
            if col not in data.columns and col in self.feature_cols:
                data[col] = data[self.target_col].shift(lag)
        
        last_date = data.index[-1]
        future_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=24, freq='h')
        
        feat_list = []
        base_prices = []
        for target_time in future_index:
            feature_time = target_time - pd.Timedelta(hours=24)
            if feature_time in data.index:
                row = data.loc[feature_time, self.feature_cols].copy()
                feat_list.append(row)
                base_prices.append(data.loc[feature_time, self.target_col])
            else:
                feat_list.append(pd.Series(0, index=self.feature_cols))
                base_prices.append(0)
        
        X_future = pd.concat(feat_list, axis=1).T.values
        X_future_scaled = self.scaler_X.transform(X_future)
        
        preds_raw = self.scaler_y.inverse_transform(self.model.predict(X_future_scaled).reshape(-1, 1)).ravel()
        
        if self.use_residuals:
            base = np.array(base_prices)
            preds_final = preds_raw + base
        else:
            preds_final = preds_raw

        if not probabilistic:
            return pd.Series(preds_final, index=future_index)

        res = pd.DataFrame(index=future_index)
        res['prediction'] = preds_final
        res['P05'] = preds_final + (self.p05_offset * self.confidence_scaling)
        res['P95'] = preds_final + (self.p95_offset * self.confidence_scaling)
        res['uncertainty'] = (res['P95'] - res['P05']) / 4
        return res
