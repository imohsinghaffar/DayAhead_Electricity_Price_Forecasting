import pandas as pd
import numpy as np
import os

class DataLoader2024:
    def __init__(self, base_dir=None):
        # Determine base directory (project root)
        if base_dir is None:
            # Assuming this file is in utils/, go up one level
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
            
    def load_data(self):
        """
        Loads and merges all data sources into a single hourly DataFrame.
        """
        print("Loading 2024 Data...")
        
        # 1. Load Price Data (GUI_ENERGY_PRICES_2024.csv)
        price_path = os.path.join(self.base_dir, 'data', 'GUI_ENERGY_PRICES_2024.csv')
        df_price = self._load_price_data(price_path)
        
        # 2. Hourly Load Data SKIPPED (User Request)
        # Relying on Price, Fuels, and Cyclical features only.
        df = df_price
        
        # 3. Load Weekly Load Data
        week_load_path = os.path.join(self.base_dir, 'LOAD_DAYAHEAD_FullYear_Data.csv')
        df = self._merge_weekly_load(df, week_load_path)
        
        # 4. Load Fuel Prices
        coal_path = os.path.join(self.base_dir, 'data', 'CoalPrices.csv')
        gas_path = os.path.join(self.base_dir, 'data', 'MonthlyGasPrice.csv')
        df = self._merge_fuel_prices(df, coal_path, gas_path)
        
        # 5. Feature Engineering (Cyclical)
        df = self._add_cyclical_features(df)
        
        # 6. Final Cleanup
        # Strict ffill for missing values (fuels/weekly) to avoid lookahead
        df = df.ffill()
        df = df.dropna() # Drop any remaining NaNs at the start
        
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df

    def _load_price_data(self, path):
        print(f"Reading Price Data: {path}")
        df = pd.read_csv(path)
        
        # Filter Area and Sequence
        # "Sequence Sequence 1" seems to be the format in the file based on head
        df = df[df['Area'] == 'BZN|DE-LU']
        df = df[df['Sequence'] == 'Sequence Sequence 1']
        
        # Parse Time (MTU UTC)
        # Format: "31/12/2023 23:00:00 - 31/12/2023 23:15:00"
        # We take the start time
        df['MTU_Start'] = df['MTU (UTC)'].str.split(' - ').str[0]
        df['Date'] = pd.to_datetime(df['MTU_Start'], format='%d/%m/%Y %H:%M:%S', utc=True)
        
        # Set Index
        df = df.set_index('Date').sort_index()
        
        # Select Price
        df = df[['Day-ahead Price (EUR/MWh)']].rename(columns={'Day-ahead Price (EUR/MWh)': 'Price'})
        
        # Convert to numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Resample to Hourly (Mean)
        # 15-min to 60-min
        df_hourly = df.resample('h').mean()
        
        return df_hourly

    def _load_hourly_load(self, path):
        print(f"Reading Hourly Load Data: {path}")
        df = pd.read_csv(path)
        
        # Filter Area if needed (file seems to have BZN|DE-LU)
        df = df[df['Area'] == 'BZN|DE-LU']
        
        # Parse Time (MTU CET/CEST)
        # Format: "01/01/2024 00:00 - 01/01/2024 00:15"
        # Parse as naive then localize? Or just trust it aligns?
        # The file says CET/CEST. 
        # "01/01/2024 00:00" CET is "31/12/2023 23:00" UTC.
        # We need to parse strict.
        
        df['MTU_Start'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]
        # Allow dayfirst=True for safety
        df['Date'] = pd.to_datetime(df['MTU_Start'], dayfirst=True)
        
        # Localize to CET then convert to UTC
        # "Europe/Berlin" handles CET/CEST transitions
        df['Date'] = df['Date'].dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward').dt.tz_convert('UTC')
        
        df = df.set_index('Date').sort_index()
        
        # Select Columns
        # "Day-ahead Total Load Forecast (MW)"
        # "Actual Total Load (MW)"
        # We focus on Forecast as usually that's what we have Day-Ahead
        cols_map = {
            'Day-ahead Total Load Forecast (MW)': 'Load_Forecast',
            'Actual Total Load (MW)': 'Load_Actual'
        }
        df = df.rename(columns=cols_map)
        
        # Keep only available columns
        available_cols = [c for c in cols_map.values() if c in df.columns]
        df = df[available_cols]
        
        # Numeric conversion
        for c in available_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Resample to Hourly (Mean)
        df_hourly = df.resample('h').mean()
        
        return df_hourly

    def _merge_weekly_load(self, df, path):
        print(f"Reading Weekly Load Data: {path}")
        df_week = pd.read_csv(path)
        
        # Extract Week Number from "Week 1", "Week 2"
        # Format: "Week X"
        df_week['Week_Num'] = df_week['Week'].str.extract(r'(\d+)').astype(int)
        
        # Rename columns
        col_map = {
            'Forecast min [MW]': 'Load_Week_Min_Forecast',
            'Forecast max [MW]': 'Load_Week_Max_Forecast'
        }
        df_week = df_week.rename(columns=col_map)
        
        # Keep only relevant
        df_week = df_week[['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast']]
        
        # Deduplicate Week_Num (take first)
        df_week = df_week.drop_duplicates(subset=['Week_Num'])
        
        # Add Week Number to main df for integration
        # Note: isoocalendar().week is standard
        df['Week_Num'] = df.index.isocalendar().week.astype(int)
        
        # Merge
        # We reset index to merge, then equality on Week_Num
        df = df.reset_index()
        df = pd.merge(df, df_week, on='Week_Num', how='left')
        df = df.set_index('Date')
        
        # Drop Week_Num if not needed, or keep? Keeping is fine.
        return df

    def _merge_fuel_prices(self, df, coal_path, gas_path):
        print("Reading Fuel Prices...")
        # Coal
        coal = pd.read_csv(coal_path)
        coal['Date'] = pd.to_datetime(coal['observation_date'])
        coal = coal.set_index('Date').sort_index()
        coal.index = coal.index.tz_localize('UTC') # Ensure UTC
        coal = coal.rename(columns={'PCOALAUUSDM': 'Coal_Price'})
        # Resample to daily/hourly to ffill
        coal = coal.resample('h').ffill()
        
        # Gas
        gas = pd.read_csv(gas_path)
        gas['Date'] = pd.to_datetime(gas['observation_date'])
        gas = gas.set_index('Date').sort_index()
        gas.index = gas.index.tz_localize('UTC') # Ensure UTC
        gas = gas.rename(columns={'PNGASEUUSDM': 'Gas_Price'})
        gas = gas.resample('h').ffill()
        
        # Merge logic:
        # Since these are monthly start dates (2024-01-01, 2024-02-01), 
        # we can merge_asof or just reindex.
        # Simplest: Reindex to df.index using ffill
        
        # We combine them into a single fuels df first
        # But ranges might differ.
        
        # Let's use reindex with ffill directly on the main df index
        coal_aligned = coal.reindex(df.index, method='ffill')
        gas_aligned = gas.reindex(df.index, method='ffill')
        
        df['Coal_Price'] = coal_aligned['Coal_Price']
        df['Gas_Price'] = gas_aligned['Gas_Price']
        
        return df

    def _add_cyclical_features(self, df):
        # Hour of day (0-23)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6)
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month of year (1-12)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # Drop raw columns
        df = df.drop(columns=['hour', 'day_of_week', 'month'])
        
        return df

if __name__ == "__main__":
    # Test run
    loader = DataLoader2024()
    df = loader.load_data()
    print(df.head())
    print(df.info())
