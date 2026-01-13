import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader2024:
    def __init__(self, data_file=None):
        """
        Initialize the 2024 data loader.
        
        Args:
            data_file: Path to the 2024 price CSV file.
        """
        self.src_dir = Path(__file__).parent.parent
        self.project_root = self.src_dir.parent
        self.data_file = Path(data_file) if data_file else self.project_root / "data" / "GUI_ENERGY_PRICES_2024.csv"
            
    def load_data(self):
        """Loads and merges all data sources into a single hourly DataFrame."""
        logger.info(f"Loading 2024 Dataset from {self.data_file}")
        
        # 1. Load Price Data
        df = self._load_price_data(self.data_file)
        
        # 2. Load Weekly Load Metadata
        week_load_path = self.project_root / "LOAD_DAYAHEAD_FullYear_Data.csv"
        df = self._merge_weekly_load(df, week_load_path)
        
        # 3. Integrate Fuel Prices (Resampled Hourly)
        df = self._merge_fuel_prices(df)
        
        # 4. Professor-required Cyclical Features
        df = self._add_cyclical_features(df)
        
        # 5. Engineering: Lags & Momentum
        df = self._add_lag_features(df)
        df = self._add_momentum_features(df)
        df = self._add_time_indicators(df)
        
        # 6. Cleanup & Alignment
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().dropna()
        
        logger.info(f"2024 Data Loaded. Shape: {df.shape}")
        return df

    def _load_price_data(self, path):
        df = pd.read_csv(path)
        logger.info(f"Raw 2024 rows: {len(df)}")
        
        # Filter: BZN|DE-LU and Sequence 1 (Robust)
        df = df[df['Area'].str.contains('DE-LU', na=False, case=False)]
        df = df[df['Sequence'].str.contains('Sequence 1', na=False, case=False)]
        logger.info(f"Rows after Area/Sequence filter: {len(df)}")
        
        if len(df) == 0:
            logger.error("No rows matching Area='DE-LU' and Sequence='Sequence 1'!")
            return pd.DataFrame()
            
        # Parse UTC Time
        df['MTU_Start'] = df['MTU (UTC)'].str.split(' - ').str[0]
        df['Date'] = pd.to_datetime(df['MTU_Start'], format='%d/%m/%Y %H:%M:%S', utc=True)
        
        df = df.set_index('Date').sort_index()
        df = df[['Day-ahead Price (EUR/MWh)']].rename(columns={'Day-ahead Price (EUR/MWh)': 'Price'})
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Resample to Hourly
        resampled = df.resample('h').mean()
        logger.info(f"Rows after resampling: {len(resampled)}")
        return resampled

    def _merge_weekly_load(self, df, path):
        if not path.exists():
            return df
        df_week = pd.read_csv(path)
        df_week['Week_Num'] = df_week['Week'].str.extract(r'(\d+)').astype(int)
        df_week = df_week.rename(columns={
            'Forecast min [MW]': 'Load_Week_Min_Forecast',
            'Forecast max [MW]': 'Load_Week_Max_Forecast'
        })[['Week_Num', 'Load_Week_Min_Forecast', 'Load_Week_Max_Forecast']].drop_duplicates(subset=['Week_Num'])
        
        df['Week_Num'] = df.index.isocalendar().week.astype(int)
        df = df.reset_index().merge(df_week, on='Week_Num', how='left').set_index('Date')
        return df

    def _merge_fuel_prices(self, df):
        """Load and align Daily Oil and Monthly Coal/Gas using merge_asof."""
        data_dir = self.project_root / "data"
        data2_dir = self.project_root / "Data2"
        
        fuel_configs = [
            (data2_dir / "DCOILBRENTEU.csv", "Oil_Price"),
            (data_dir / "CoalPrices.csv", "Coal_Price"),
            (data_dir / "MonthlyGasPrice.csv", "Gas_Price")
        ]
        
        for path, col_name in fuel_configs:
            if not path.exists():
                logger.warning(f"Fuel file not found: {path}")
                continue
                
            f_df = pd.read_csv(path)
            f_df['Date'] = pd.to_datetime(f_df.iloc[:, 0], utc=True)
            f_df = f_df.sort_values('Date')
            
            # Ensure numeric
            val_col = f_df.columns[1] if 'Date' in f_df.columns else f_df.columns[0]
            f_df[col_name] = pd.to_numeric(f_df[val_col], errors='coerce')
            f_df = f_df[['Date', col_name]].dropna()
            
            # Asof merge to align to hourly index
            df = df.reset_index()
            df = pd.merge_asof(
                df.sort_values('Date'), 
                f_df, 
                on='Date', 
                direction='backward'
            ).set_index('Date')
            
            non_nan = df[col_name].notna().sum()
            logger.info(f"Integrated fuel {col_name}. Matches: {non_nan}/{len(df)}")
            
        return df

    def _add_cyclical_features(self, df):
        """Professor-mandated cyclical feature names."""
        # Hour of the day
        hour = df.index.hour
        df['hour_of_the_day_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_of_the_day_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of the week
        dow = df.index.dayofweek
        df['day_of_the_week_sin'] = np.sin(2 * np.pi * dow / 7)
        df['day_of_the_week_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Month of the year
        month = df.index.month
        df['month_of_the_year_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df['month_of_the_year_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        return df
    
    def _add_lag_features(self, df):
        df['price_lag_1'] = df['Price'].shift(1)
        df['price_lag_24'] = df['Price'].shift(24)
        df['price_lag_48'] = df['Price'].shift(48)
        df['price_lag_168'] = df['Price'].shift(168)
        
        shift_1 = df['Price'].shift(1)
        df['price_rolling_mean_24'] = shift_1.rolling(window=24, min_periods=1).mean()
        df['price_rolling_std_24'] = shift_1.rolling(window=24, min_periods=1).std()
        df['price_rolling_min_24'] = shift_1.rolling(window=24, min_periods=1).min()
        df['price_rolling_max_24'] = shift_1.rolling(window=24, min_periods=1).max()
        return df
    
    def _add_momentum_features(self, df):
        df['price_diff_24'] = df['Price'] - df['Price'].shift(24)
        df['price_change_pct'] = df['Price'].pct_change(24) * 100
        shift_1 = df['Price'].shift(1)
        df['volatility_24h'] = shift_1.rolling(window=24, min_periods=1).std()
        df['volatility_168h'] = shift_1.rolling(window=168, min_periods=1).std()
        df['price_range_24h'] = df['price_rolling_max_24'] - df['price_rolling_min_24']
        return df
    
    def _add_time_indicators(self, df):
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        hour = df.index.hour
        df['is_peak_hour'] = ((hour >= 7) & (hour <= 10) | (hour >= 17) & (hour <= 21)).astype(int)
        df['is_business_hours'] = ((hour >= 9) & (hour <= 17) & (df.index.dayofweek < 5)).astype(int)
        return df

if __name__ == "__main__":
    loader = DataLoader2024()
    df = loader.load_data()
    print(df.head())
