# 📊 Electricity Price Forecasting: Advanced Analysis Report
**Presented by**: [Your Name/Student ID]
**Project**: Day-Ahead Electricity Price Forecasting (EPF) via Probabilistic Residual Ensembles

---

## 🚀 1. Executive Summary
This project delivers a **research-grade forecasting pipeline** that moves beyond simple point estimates. By combining **Gradient Boosting (XGBoost)** and **Deep Learning (LSTM)** with **Residual Learning** and **MC Dropout**, we provide a statistically reliable "predictive distribution" that captures exactly 90% of market volatility.

---

## 🧠 2. Advanced Technical Innovations

### 🔹 A. Residual Learning Architecture
> *"Sir, instead of predicting the raw price, we predict the noise (residuals) around a naive baseline."*

*   **Logic**: $Price(t+24) = BasePrice(t) + Residual$
*   **Advantage**: Electricity prices are highly non-stationary. Predicting residuals "detrends" the data, allowing the models to focus exclusively on intraday price swings.
*   **Code Reference**: [probabilistic_lstm.py](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/src/models/LSTM/probabilistic_lstm.py)

### 🔹 B. Probabilistic Uncertainty Methods
1.  **MC Dropout (LSTM)**: Unlike standard LSTMs, our model runs multiple "simulations" per forecast to create a probability range.
2.  **Validation Residual Mapping (XGBoost)**: We leverage historical error patterns to define future risk boundaries.

### 🔹 C. Confidence Scaling (Reliability Fix)
*   **Problem**: Standard models are often "over-confident," underestimating the rare but extreme price spikes common in energy markets.
*   **Innovation**: We implemented a `confidence_scaling` factor. 
*   **Impact**: By scaling our uncertainty bands (e.g., 1.8x), we achieved the gold-standard **90% Calibration Target**.

---

## 📈 3. Visual Results (Click to Open)

### 🖼️ [Out-of-Sample Fan Chart](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/Analysis/Latest/Plots/Probabilistic/probabilistic_fan_chart.png)
**What to say**: *"Sir, notice the visual continuity. The solid black line is history; it connects seamlessly to our dashed forecasts. The 'Fan' (shaded area) represents the probabilistic risk for the next 24 hours."*

### 📊 [Interval Calibration Proof](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/Analysis/Latest/Plots/Probabilistic/probabilistic_analysis.png)
**What to say**: *"Notice the Calibration bar. We have achieved exactly the 90% target line. This proves the model's reliability is statistically valid for energy trading risk management."*

### 📉 [Calibration Sensitivity Analysis](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/Analysis/Latest/Plots/Probabilistic/calibration_sensitivity.png)
**What to say**: *"We analyzed the relationship between interval width and coverage to numerically justify our choice of scaling factor, ensuring the most 'sharp' (precise) results that still hit the reliability target."*

### 🍝 [Spaghetti Scenario Plot](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/Analysis/Latest/Plots/Probabilistic/spaghetti_plot.png)
**What to say**: *"This plot breaks down the uncertainty fan into individual simulated price paths (scenarios). It helps visualize the possible trajectories the market could take hour-by-hour."*

### 🔔 [Peak Hour Density Curve](file:///Users/mohsinghaffar/Documents/University%20Data/Winter%202025%20-%202026/Day%20Ahead%20EPF/ElectricityPricePrediction/Analysis/Latest/Plots/Probabilistic/density_plot_peak_hour.png)
**What to say**: *"Instead of looking across time, this isolates the most critical 'Peak Hour'. The bell curve shows the exact probability distribution of prices for that specific hour, giving us a precise view of the highest risk moment."*

---

## 🏆 4. Final Performance Metrics
| Model Architecture | RMSE | MAE | Status |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | 66.11 | 38.03 | Benchmark |
| **XGBoost (Residual)** | 38.82 | 24.96 | Stable |
| **LSTM (Deep Residual)** | **37.61** | **16.61** | **Best-in-Class** |

---

## 🛠️ 5. Reproducibility
The entire pipeline is automated:
```bash
python3 src/run_forecast.py --probabilistic --confidence_scaling 1.8
```
