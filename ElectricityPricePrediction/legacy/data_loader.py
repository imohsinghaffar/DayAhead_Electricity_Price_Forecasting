import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        # Flexible path handling
        self.data_dir = data_dir

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cyclical features exactly as requested by the professor.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex.")

        df = df.copy()

        # 1. Hour of the day
        df['hour_of_the_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_of_the_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # 2. Day of the week
        df['day_of_the_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_the_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # 3. Month of the year
        # (Month is 1-12, so we subtract 1 to make it 0-11 for the cycle)
        month_idx = df.index.month - 1
        df['month_of_the_year_sin'] = np.sin(2 * np.pi * month_idx / 12)
        df['month_of_the_year_cos'] = np.cos(2 * np.pi * month_idx / 12)

        return df

    def load_data(self, filename='re_fixed_multivariate_timeseires.csv') -> pd.DataFrame:
        """
        Loads data, fills gaps, and adds features.
        """
        # Search for file in common locations
        paths = [
            os.path.join(self.data_dir, filename),
            os.path.join('ElectricityPricePrediction', 'data', filename),
            filename
        ]
        
        file_path = None
        for p in paths:
            if os.path.exists(p):
                file_path = p
                break
        
        if not file_path:
            # Fallback: Try to find any csv if the specific name isn't found
            raise FileNotFoundError(f"File {filename} not found in search paths.")

        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Smart Date Parsing
        time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not time_cols:
            raise ValueError("No datetime column found.")
        
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
        df.set_index(time_cols[0], inplace=True)
        df.sort_index(inplace=True)
        
        # Clean Data (Fill NaNs)
        df = df.ffill().bfill()

        # Add Features
        df = self.add_cyclical_features(df)
        
        return df