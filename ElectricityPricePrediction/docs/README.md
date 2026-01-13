# Day Ahead Electricity Price Prediction

This project aims to predict the price of electricity in day-ahead markets 24 hours in advance using various machine learning models.

#### Project Status: [Completed/Refactored]

## Project Structure

The project is organized into modular components for easier navigation and reproducibility:

```
ElectricityPricePrediction/
├── run_forecast.py           # Main entry point to run the pipeline
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
├── utils/
│   └── data_loader.py        # Data loading and feature engineering (Cyclical features)
├── Naive/
│   ├── naive_baselines.py    # Naive baseline models (24h, 7d persistence)
│   └── ...                   # Original notebooks
├── XGBoost/
│   ├── xgboost_model.py      # XGBoost implementation
│   └── ...                   # Original notebooks
├── LSTM/
│   ├── lstm_model.py         # LSTM (Deep Learning) implementation
│   └── ...
└── Analysis/
    ├── impact_analysis.py    # Feature importance analysis
    └── model_comparison.py   # Legacy comparison script
```

## Features
The input data includes:
* **Hourly Electricity Prices** (GBP/mWh)
* **Commodity Prices**: Coal, Natural Gas, Uranium, Oil
* **Weather Data**: Temperature
* **Derived Cyclical Features**: 
    * Hour of Day (Sin/Cos)
    * Day of Week (Sin/Cos)
    * Month of Year (Sin/Cos)

## Models
Three primary model types are implemented and compared:
1. **Naive Baselines**:
   - `Naive 24h`: Persists the price from the same hour yesterday.
   - `Naive 7d`: Persists the price from the same hour last week.
2. **XGBoost**: Gradient boosted decision trees using all available features and lag variables.
3. **LSTM**: Long Short-Term Memory neural network for sequence prediction (requires TensorFlow).

## Getting Started

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone <repo_url>
cd ElectricityPricePrediction
pip install -r requirements.txt
```
*Note: TensorFlow is optional but required for the LSTM model. If not installed, the pipeline will skip LSTM.*

### 2. Running the Pipeline
To run the full forecasting and comparison pipeline, simply execute the main script:

```bash
python run_forecast.py
```

> [!WARNING]
> **Mac M-Series Users**: If you encounter a "segmentation fault" when running the above, it is due to an incompatibility with your installed TensorFlow version and your sophisticated M-chip. This is a known issue.
> 
> **Solution**: Run the pipeline without the LSTM model:
> ```bash
> python run_forecast.py --skip_lstm
> ```

This will:
1. Load and preprocess the data.
2. Train and evaluate all available models.
3. Print a performance summary (RMSE/MAE/MAPE).
4. Save a results text file and a forecast comparison plot in `Analysis/plots/`.

### 3. Feature Impact Analysis
To see which features (e.g., Oil Price vs. Coal Price) drive the model the most, run:
```bash
python Analysis/impact_analysis.py
```

## Contact
* Original Project by: Carter Bouley
* Contributed by: Mohsin Ghaffar

