import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class NaiveBaseline:
    def __init__(self, df, target_col='GBP/mWh'):
        """
        Initialize the NaiveBaseline model.
        
        Args:
            df (pd.DataFrame): The dataframe containing the time series data.
                               Must have a datetime index.
            target_col (str): The name of the target column (price).
        """
        self.df = df.copy()
        self.target_col = target_col

    def predict_24h(self):
        """
        Predicts the price using a 24-hour lag (1 day).
        """
        self.df['pred_24h'] = self.df[self.target_col].shift(24)
        return self.df['pred_24h']

    def predict_7d(self):
        """
        Predicts the price using a 168-hour lag (7 days).
        """
        self.df['pred_7d'] = self.df[self.target_col].shift(168)
        return self.df['pred_7d']

    def evaluate(self, test_start_index, prediction_col):
        """
        Evaluates the model on a test set.
        
        Args:
            test_start_index (int or datetime): The start index for the test set.
            prediction_col (str): The column name of the predictions to evaluate.
            
        Returns:
            dict: A dictionary containing RMSE and MAE.
        """
        # Drop NaNs created by shifting
        test_df = self.df.loc[test_start_index:].dropna(subset=[self.target_col, prediction_col])
        
        if test_df.empty:
            return {'rmse': np.nan, 'mae': np.nan}

        y_true = test_df[self.target_col]
        y_pred = test_df[prediction_col]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {'rmse': rmse, 'mae': mae}

    def plot_comparison(self, start_idx, end_idx, prediction_cols=['pred_24h', 'pred_7d']):
        """
        Plots actual vs predicted values for a specific range.
        """
        subset = self.df.loc[start_idx:end_idx]
        
        plt.figure(figsize=(15, 7))
        plt.plot(subset.index, subset[self.target_col], label='Actual', color='black')
        
        if 'pred_24h' in prediction_cols and 'pred_24h' in subset.columns:
            plt.plot(subset.index, subset['pred_24h'], label='24h Naive', linestyle='--')
            
        if 'pred_7d' in prediction_cols and 'pred_7d' in subset.columns:
            plt.plot(subset.index, subset['pred_7d'], label='7d Naive', linestyle='-.')
            
        plt.title('Naive Baselines vs Actual')
        plt.legend()
        plt.show()
