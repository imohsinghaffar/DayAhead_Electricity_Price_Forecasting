import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir='ElectricityPricePrediction/data'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Path to the directory containing data files.
        """
        self.data_dir = data_dir

    def load_data(self, price_file='germany_price_2024.csv', load_file='germany_load_2024.csv'):
        """
        Loads and merges the price and load data.
        
        Args:
            price_file (str): Filename for price data.
            load_file (str): Filename for load data.
            
        Returns:
            pd.DataFrame: Merged dataframe with datetime index.
        """
        price_path = os.path.join(self.data_dir, price_file)
        load_path = os.path.join(self.data_dir, load_file)
        
        if not os.path.exists(price_path):
            raise FileNotFoundError(f"Price data file not found at {price_path}. Please download the 2024 data.")
            
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load data file not found at {load_path}. Please download the 2024 data.")

        print(f"Loading price data from {price_path}...")
        # Assuming standard ENTSO-E format or similar CSV structure
        # We might need to adjust read_csv parameters (sep, header, etc.) based on actual file
        try:
            df_price = pd.read_csv(price_path)
            # Basic preprocessing assumption: 'Date' or 'Time' column exists
            # We'll try to identify datetime column automatically or assume standard names
            time_col_price = [c for c in df_price.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            df_price[time_col_price] = pd.to_datetime(df_price[time_col_price])
            df_price.set_index(time_col_price, inplace=True)
            # Rename price column to standard 'GBP/mWh' or 'Price'
            # For now, let's assume we rename the first numeric column to 'Price'
            price_col = [c for c in df_price.columns if df_price[c].dtype in ['float64', 'int64']][0]
            df_price.rename(columns={price_col: 'Price'}, inplace=True)
        except Exception as e:
            raise ValueError(f"Error processing price file: {e}")

        print(f"Loading load data from {load_path}...")
        try:
            df_load = pd.read_csv(load_path)
            time_col_load = [c for c in df_load.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            df_load[time_col_load] = pd.to_datetime(df_load[time_col_load])
            df_load.set_index(time_col_load, inplace=True)
            # Rename load column
            load_col = [c for c in df_load.columns if df_load[c].dtype in ['float64', 'int64']][0]
            df_load.rename(columns={load_col: 'LoadForecast'}, inplace=True)
        except Exception as e:
            raise ValueError(f"Error processing load file: {e}")

        # Merge
        print("Merging datasets...")
        df_merged = df_price.join(df_load, how='inner')
        
        # Sort index
        df_merged.sort_index(inplace=True)
        
        # Handle missing values
        df_merged.fillna(method='ffill', inplace=True)
        
        return df_merged
