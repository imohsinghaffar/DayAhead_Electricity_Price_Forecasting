# Electricity Price Forecasting Pipeline (German Market)

A comprehensive machine learning pipeline for day-ahead electricity price forecasting in Germany (BZN|DE-LU Area).

## 📁 Project Structure (A–Z Refactor)
```
ElectricityPricePrediction/
├── data/                    # Unified data directory
│   ├── GUI_ENERGY_PRICES_2024.csv
│   ├── weather/             # DWD station data
│   └── fuel_prices/         # Oil, Coal, Gas
├── src/                     # Source code
│   ├── run_forecast.py      # Entrypoint
│   ├── models/              # Naive, XGBoost, LSTM
│   └── utils/               # Loaders & Visualizers
├── results/                 # CSV outputs & Metrics
└── Analysis/                # 300 DPI plots & latest stats
```

## 🧪 Key Features
- **Professor-Mandated Cyclical Encoding**: `hour_of_the_day_sin/cos`, `day_of_the_week_sin/cos`, `month_of_the_year_sin/cos`.
- **Probabilistic Forecasting**: 3-layer LSTM with Monte Carlo Dropout for uncertainty estimation.
- **Robust Integration**: Hourly resampling for Daily Oil and Monthly Coal/Gas fuel prices.
- **Weather Station Integration**: Aggregated data from 8 major German cities via DWD API.
- **Reproducibility**: Global seed management and standard `ggplot` style visualizations at 300 DPI.

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Standard Run (2024 Data)
```bash
python src/run_forecast.py --use_weather
```

### 3. Historical Run (2019–2024)
```bash
python src/run_forecast.py --use_historical --use_weather --visualize_optuna
```

### 4. Probabilistic Run (Uncertainty)
```bash
python src/run_forecast.py --probabilistic --use_weather
```

## 📊 Results & Analysis
- **Metrics**: `results/csv/results_metrics.csv`
- **Forecasts**: `results/forecast_values.csv` (includes uncertainty column if --probabilistic)
- **Visualizations**: `Analysis/plots/comprehensive_analysis.png` (300 DPI)

### 📈 Interactive Dashboard (View in Browser)
Use the links below to view interactive charts (Zoom, Hover, Pan) directly without downloading:
- 🔗 [**Live Forecast Comparison**](https://htmlpreview.github.io/?https://github.com/imohsinghaffar/DayAhead_Electricity_Price_Forecasting/blob/main/ElectricityPricePrediction/Analysis/Latest/Plots/Interactive/interactive_forecast.html)
- 🔗 [**Interactive Error Analysis**](https://htmlpreview.github.io/?https://github.com/imohsinghaffar/DayAhead_Electricity_Price_Forecasting/blob/main/ElectricityPricePrediction/Analysis/Latest/Plots/Interactive/interactive_error_analysis.html)
- 🔗 [**Latest Training Report (HTML)**](https://htmlpreview.github.io/?https://github.com/imohsinghaffar/DayAhead_Electricity_Price_Forecasting/blob/main/ElectricityPricePrediction/Analysis/Latest/report.html)


## ✍️ Author
Electricity Price Prediction Project Refactor
