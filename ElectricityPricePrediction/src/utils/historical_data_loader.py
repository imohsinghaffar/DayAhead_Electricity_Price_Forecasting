import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    """
    Load and merge 2019-2024 electricity price data.
    Aligns fuel prices (Daily Oil, Monthly Coal) to hourly index.
    """
    
    def __init__(self, data_dir=None):
        self.src_dir = Path(__file__).parent.parent
        self.project_root = self.src_dir.parent
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "Data2"
        self.data_dir_2024 = self.project_root / "data"
        
    def load_full_dataset(self):
        """Main method: Load 2019-2023 archive + 2024 recent data."""
        logger.info("Loading full historical range 2019-2024...")
        
        # 1. Archive Data (2019-2023)
        archive_df = self._load_archive_data()
        
        # 2. Recent Data (2024)
        from utils.data_loader_2024 import DataLoader2024
        loader_24 = DataLoader2024(data_file=self.data_dir_2024 / "GUI_ENERGY_PRICES_2024.csv")
        recent_df = loader_24.load_data()
        
        # Combine (archive has only Price initially, recent_df has features)
        # We only need 'Price' from both to start features from scratch OR align manually
        # Cleaner: Merge Prices first, then add features globally to ensure consistency
        
        full_price_df = pd.concat([
            archive_df[['Price']], 
            recent_df[['Price']]
        ]).sort_index()
        
        # Remove duplicates from overlap
        full_price_df = full_price_df[~full_price_df.index.duplicated(keep='last')]
        
        # 3. Features (Calculated on Full Range)
        df = self._add_cyclical_features(full_price_df)
        df = self._merge_fuel_prices(df)
        df = self._add_lag_features(df)
        df = self._add_momentum_features(df)
        df = self._add_time_indicators(df)
        
        # Cleanup
        df = df.ffill().dropna()
        
        logger.info(f"Historical 2019-2024 Loaded. Shape: {df.shape}")
        return df

    def _load_archive_data(self):
        yearly_files = [
            "GUI_ENERGY_PRICES_201901010000-202001010000.csv",
            "GUI_ENERGY_PRICES_202001010000-202101010000.csv",
            "GUI_ENERGY_PRICES_202101010000-202201010000.csv",
            "GUI_ENERGY_PRICES_202201010000-202301010000.csv",
            "GUI_ENERGY_PRICES_202301010000-202401010000.csv"
        ]
        all_data = []
        for file in yearly_files:
            p = self.data_dir / file
            if p.exists():
                df = pd.read_csv(p)
                df = self._parse_generic_entsoe(df)
                all_data.append(df)
        
        return pd.concat(all_data).sort_index()

    def _parse_generic_entsoe(self, df):
        # Auto-detect columns
        time_col = [c for c in df.columns if any(x in c.lower() for x in ['time', 'mtu', 'date'])][0]
        price_col = [c for c in df.columns if any(x in c.lower() for x in ['price', 'day ahead'])][0]
        
        df['Date'] = pd.to_datetime(df[time_col].str.split(' - ').str[0], utc=True)
        df = df.set_index('Date').sort_index()
        df = df[[price_col]].rename(columns={price_col: 'Price'})
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        return df.resample('h').mean()

    def _merge_fuel_prices(self, df):
        """Align Daily Oil and Monthly Coal/Gas using merge_asof."""
        fuel_configs = [
            (self.data_dir / "DCOILBRENTEU.csv", "Oil_Price"),
            (self.data_dir / "PCOALAUUSDM.csv", "Coal_Price"),
            (self.data_dir_2024 / "TTF_Natural_Gas.csv", "Gas_Price")
        ]
        
        for path, col_name in fuel_configs:
            if not path.exists():
                logger.warning(f"Historical Fuel file not found: {path}")
                continue
                
            f_df = pd.read_csv(path)
            f_df['Date'] = pd.to_datetime(f_df.iloc[:,0], utc=True)
            f_df = f_df.sort_values('Date')
            
            val_col = f_df.columns[1] if 'Date' in f_df.columns else f_df.columns[0]
            f_df[col_name] = pd.to_numeric(f_df[val_col], errors='coerce')
            f_df = f_df[['Date', col_name]].dropna()
            
            df = df.reset_index()
            df = pd.merge_asof(
                df.sort_values('Date'),
                f_df,
                on='Date',
                direction='backward'
            ).set_index('Date')
            
            logger.info(f"Historical {col_name} integrated. Matches: {df[col_name].notna().sum()}/{len(df)}")
            
        return df

    def _add_cyclical_features(self, df):
        hour = df.index.hour
        df['hour_of_the_day_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_of_the_day_cos'] = np.cos(2 * np.pi * hour / 24)
        dow = df.index.dayofweek
        df['day_of_the_week_sin'] = np.sin(2 * np.pi * dow / 7)
        df['day_of_the_week_cos'] = np.cos(2 * np.pi * dow / 7)
        month = df.index.month
        df['month_of_the_year_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df['month_of_the_year_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        return df

    def _add_lag_features(self, df):
        df['price_lag_1'] = df['Price'].shift(1)
        df['price_lag_24'] = df['Price'].shift(24)
        df['price_lag_168'] = df['Price'].shift(168)
        return df

    def _add_momentum_features(self, df):
        shift_1 = df['Price'].shift(1)
        df['price_rolling_mean_24'] = shift_1.rolling(window=24).mean()
        df['volatility_24h'] = shift_1.rolling(window=24).std()
        return df

    def _add_time_indicators(self, df):
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        return df
