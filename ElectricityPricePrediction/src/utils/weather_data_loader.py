import pandas as pd
import numpy as np
import logging
import requests
import zipfile
import io
from pathlib import Path

logger = logging.getLogger(__name__)

STATIONS = {
    '00433': 'Berlin-Tempelhof', '01262': 'Dresden-Klotzsche', 
    '01975': 'Frankfurt-Main', '02014': 'Hannover',
    '02667': 'Koeln-Bonn', '03379': 'Muenchen-Stadt',
    '01228': 'Hamburg-Fuhlsbuettel', '04931': 'Stuttgart-Echterdingen'
}

DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature"

class WeatherDataLoader:
    def __init__(self, data_dir=None):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data" / "weather"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_station_zip(self, station_id, mode="recent"):
        # Historical zip names are different (stundenwerte_TU_00433_19480101_20231231_hist.zip)
        # For simplicity, we use the 'akt' for recent and try to find hist file if needed.
        # However, DWD recent is usually last ~1.5 years.
        
        url_mode = "recent" if mode == "recent" else "historical"
        suffix = "akt.zip" if mode == "recent" else "hist.zip"
        
        # Searching for historical filename is tricky because of date ranges.
        # We'll use a simplified approach: Download recent, and if historical requested, 
        # listing the directory first would be better.
        
        if mode == "recent":
            url = f"{DWD_BASE}/recent/stundenwerte_TU_{station_id}_akt.zip"
        else:
            # For historical, we'd ideally scrape the directory.
            # Fallback: Many stations have a predictable hist name prefix.
            return None # Placeholder for complex scraping if needed

        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return self._extract_dwd_zip(resp.content, station_id)
        except Exception as e:
            logger.error(f"DWD Download Error ({station_id}): {e}")
        return None

    def _extract_dwd_zip(self, content, station_id):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            data_file = [f for f in z.namelist() if f.startswith('produkt_')][0]
            with z.open(data_file) as f:
                df = pd.read_csv(f, sep=';', na_values=['-999', '-999.0'])
        df = df.rename(columns={'MESS_DATUM': 'Date', 'TT_TU': f'temp_{station_id}'})
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H', utc=True)
        return df.set_index('Date')[[f'temp_{station_id}']]

    def get_weather_features(self, start_date, end_date):
        logger.info(f"Fetching DWD Weather (8 stations) for {start_date} to {end_date}...")
        all_dfs = []
        for sid in STATIONS:
            # Try recent
            df_recent = self.download_station_zip(sid, "recent")
            if df_recent is not None:
                all_dfs.append(df_recent)
        
        if not all_dfs:
            return self._load_fallback()

        merged = pd.concat(all_dfs, axis=1)
        merged = merged[~merged.index.duplicated(keep='last')]
        
        # Aggregates
        merged['temp_avg'] = merged.mean(axis=1)
        merged['temp_min'] = merged.min(axis=1)
        merged['temp_max'] = merged.max(axis=1)
        merged['temp_std'] = merged.std(axis=1)
        
        # Lags
        merged['temp_lag_24'] = merged['temp_avg'].shift(24)
        merged['temp_diff_24'] = merged['temp_avg'] - merged['temp_lag_24']
        merged['temp_rolling_mean_24'] = merged['temp_avg'].rolling(24).mean()
        
        return merged.loc[start_date:end_date].ffill()

    def _load_fallback(self):
        fallback = self.project_root / "data" / "weather_file.csv"
        if fallback.exists():
            df = pd.read_csv(fallback, header=None, names=['Date', 'temp_avg'])
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.set_index('Date').resample('h').mean().ffill()
            return df
        return None

def merge_weather_with_prices(price_df, weather_df):
    cols = ['temp_avg', 'temp_min', 'temp_max', 'temp_std', 'temp_lag_24', 'temp_diff_24', 'temp_rolling_mean_24']
    valid_cols = [c for c in cols if c in weather_df.columns]
    return price_df.join(weather_df[valid_cols], how='left').ffill().bfill()
