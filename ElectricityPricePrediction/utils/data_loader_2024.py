import pandas as pd
import numpy as np
import os

class DataLoader2024:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.price_file = os.path.join(data_dir, 'GUI_ENERGY_PRICES_2024.csv')
        self.coal_file = os.path.join(data_dir, 'CoalPrices.csv')
        self.gas_file = os.path.join(data_dir, 'MonthlyGasPrice.csv')

    def load_data(self):
        """
        Loads and processes the 2024 Market Data and Commodities.
        Returns a cleaned DataFrame with hourly frequency.
        """
        print(f"Loading 2024 Data from {self.data_dir}...")
        
        # 1. Load Electricity Prices
        # Structure: "MTU (UTC)","Area","Sequence","Day-ahead Price (EUR/MWh)",...
        # Timestamp format: "31/12/2023 23:00:00 - 31/12/2023 23:15:00" (Quarter hourly?)
        # Wait, the rows show 15 min intervals e.g. 23:00-23:15. 
        # But user asked for hourly comparison. I might need to resample or check if hourly data exists.
        # The prompt says "with hourly timesteps". The file has 15 min resolution.
        # I will resample to Hourly Mean.
        
        df_price = pd.read_csv(self.price_file)
        
        # Filter Sequence 2
        df_price = df_price[df_price['Sequence'] == 'Sequence Sequence 2'].copy()
        
        # Parse Dates (Take the start part of "Start - End")
        # "31/12/2023 23:00:00 - ..." -> "31/12/2023 23:00:00"
        df_price['timestamp_str'] = df_price['MTU (UTC)'].str.split(' - ').str[0]
        df_price['Date'] = pd.to_datetime(df_price['timestamp_str'], format='%d/%m/%Y %H:%M:%S')
        
        # Set Index
        df_price.set_index('Date', inplace=True)
        df_price.sort_index(inplace=True)
        
        # Select Price Column
        price_col = 'Day-ahead Price (EUR/MWh)'
        if price_col not in df_price.columns:
            raise ValueError(f"Price column '{price_col}' not found.")
            
        df_price = df_price[[price_col]].rename(columns={price_col: 'Price'})
        
        # Convert to numeric (handle potential string issues)
        df_price['Price'] = pd.to_numeric(df_price['Price'], errors='coerce')
        
        # Resample to Hourly (Taking mean of the 4 quarters)
        df_hourly = df_price.resample('h').mean()
        
        # Interpolate small gaps if any
        df_hourly['Price'] = df_hourly['Price'].interpolate(method='linear')

        # 2. Load Commodities (Monthly) - Restored
        # Coal
        coal_df = pd.read_csv(self.coal_file)
        coal_df['observation_date'] = pd.to_datetime(coal_df['observation_date'])
        coal_df.set_index('observation_date', inplace=True)
        coal_df.rename(columns={'PCOALAUUSDM': 'Coal_Price'}, inplace=True)
        
        # Gas
        gas_df = pd.read_csv(self.gas_file)
        gas_df['observation_date'] = pd.to_datetime(gas_df['observation_date'])
        gas_df.set_index('observation_date', inplace=True)
        gas_df.rename(columns={'PNGASEUUSDM': 'Gas_Price'}, inplace=True)
        
        # 3. Merge Commodities
        def upsample_monthly(monthly_df, target_idx):
            combined_idx = monthly_df.index.union(target_idx).sort_values()
            ts = monthly_df.reindex(combined_idx)
            ts = ts.ffill() 
            return ts.reindex(target_idx)
            
        df_hourly['Coal'] = upsample_monthly(coal_df, df_hourly.index)['Coal_Price']
        df_hourly['Gas'] = upsample_monthly(gas_df, df_hourly.index)['Gas_Price']
        
        # 4. Fetch Weather Data (Open-Meteo API)
        # Using Berlin coords (52.52, 13.41) as Germany proxy
        try:
            print("Fetching Weather Data from Open-Meteo (Berlin)...")
            # 2024 full year
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": 52.52,
                "longitude": 13.41,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "hourly": "temperature_2m,wind_speed_10m"
            }
            import requests
            response = requests.get(url, params=params)
            data = response.json()
            
            weather_hourly = data['hourly']
            df_weather = pd.DataFrame({
                'Date': pd.to_datetime(weather_hourly['time']),
                'Temperature': weather_hourly['temperature_2m'],
                'Wind_Speed': weather_hourly['wind_speed_10m']
            })
            df_weather.set_index('Date', inplace=True)
            
            # Merge Weather (Left join/Reindex to ensure alignment)
            # Weather is hourly, matching price
            df_hourly = df_hourly.join(df_weather, how='left')
            
            # Fill small gaps if API missing
            df_hourly['Temperature'] = df_hourly['Temperature'].interpolate()
            df_hourly['Wind_Speed'] = df_hourly['Wind_Speed'].interpolate()
            
        except Exception as e:
            print(f"Warning: Failed to fetch weather data: {e}. Proceeding without it.")
        
        # Fill any remaining NaNs (e.g. if price data starts before commodity data)
        df_hourly.ffill(inplace=True)
        df_hourly.bfill(inplace=True)
        
        # 5. Feature Engineering (Cyclical)
        df_hourly = self.add_cyclical_features(df_hourly)
        
        # 6. Add Lag Features (Crucial for LSTM to beat Naive)
        # Explicitly give "Yesterday's Price" and "Last Week's Price"
        df_hourly['lag_24h'] = df_hourly['Price'].shift(24)
        df_hourly['lag_168h'] = df_hourly['Price'].shift(168)
        
        # 7. Residual Target (Price - Lag24)
        # We will try to predict the *change* from yesterday, not the absolute price.
        df_hourly['Residual'] = df_hourly['Price'] - df_hourly['lag_24h']
        
        # Drop NaNs created by lags
        df_hourly = df_hourly.dropna()

        print(f"Data Loaded. Shape: {df_hourly.shape}")
        return df_hourly

    def add_cyclical_features(self, df):
        df = df.copy()
        
        # Hour
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of Week
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # --- Rolling Features (Volatility & Trends) ---
        # 24h (Daily)
        df['rolling_mean_24h'] = df['Price'].rolling(window=24, min_periods=1).mean()
        df['rolling_std_24h']  = df['Price'].rolling(window=24, min_periods=1).std().fillna(0)
        
        # 168h (Weekly)
        df['rolling_mean_168h'] = df['Price'].rolling(window=168, min_periods=1).mean()
        df['rolling_std_168h'] = df['Price'].rolling(window=168, min_periods=1).std().fillna(0)
        
        # 720h (Monthly)
        df['rolling_mean_720h'] = df['Price'].rolling(window=720, min_periods=1).mean()
        
        return df

if __name__ == "__main__":
    loader = DataLoader2024()
    df = loader.load_data()
    print(df.head())
    print(df.describe())
