import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class XGBoostModel:
    def __init__(self, df, target_col='GBP/mWh', feature_cols=None):
        """
        Initialize the XGBoostModel.
        
        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
            target_col (str): The name of the target column.
            feature_cols (list): List of feature column names. If None, uses all columns except target.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [c for c in df.columns if c != target_col]
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        """
        Scales the input features and target variable.
        Adds lag features to capture autoregressive properties.
        """
        # Feature Engineering: Add Lags
        # Ensure we don't overwrite if they exist, but here we enforce them for better performance
        for lag in [24, 48, 168]:
            col_name = f"{self.target_col}_lag_{lag}"
            if col_name not in self.df.columns:
                self.df[col_name] = self.df[self.target_col].shift(lag)
                if col_name not in self.feature_cols:
                    self.feature_cols.append(col_name)

        # Handle missing values created by shifting
        self.df.fillna(method='ffill', inplace=True)
        # Drop initial rows with NaNs from lags (optional, or fillna handles it)
        self.df.dropna(inplace=True)
        
        X = self.df[self.feature_cols].values
        y = self.df[[self.target_col]].values

        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)
        
        return self.X_scaled, self.y_scaled

    def train(self, train_end_idx, params=None):
        """
        Trains the XGBoost model.
        
        Args:
            train_end_idx (int): Index to split training and validation data.
            params (dict): XGBoost hyperparameters.
        """
        if params is None:
            # Default parameters from previous notebook or reasonable defaults
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1
            }

        X_train = self.X_scaled[:train_end_idx]
        y_train = self.y_scaled[:train_end_idx]
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        
        print("XGBoost training completed.")

    def predict(self, test_start_idx):
        """
        Generates predictions for the test set.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet.")

        X_test = self.X_scaled[test_start_idx:]
        
        if len(X_test) == 0:
            return np.array([])

        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        return y_pred.flatten()

    def evaluate(self, test_start_idx):
        """
        Evaluates the model on the test set.
        """
        y_pred = self.predict(test_start_idx)
        y_true = self.df[self.target_col].values[test_start_idx:]
        
        # Align lengths in case of mismatch (though shouldn't happen with logic above)
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {'rmse': rmse, 'mae': mae, 'y_pred': y_pred, 'y_true': y_true}
