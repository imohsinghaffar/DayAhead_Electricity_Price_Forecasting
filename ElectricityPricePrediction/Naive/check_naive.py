import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# 1) Load your existing dataset
# ----------------------------
# Fixed path: ../data/re_fixed_multivariate_timeseires.csv
df = pd.read_csv("../data/re_fixed_multivariate_timeseires.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.drop_duplicates("datetime").set_index("datetime").sort_index()

# unify price col name
if "GBP/mWh" in df.columns and "Price" not in df.columns:
    df = df.rename(columns={"GBP/mWh":"Price"})

# ----------------------------
# 2) Define helper metrics
# ----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    # Avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

# ----------------------------
# 3) Create Naïve baselines (point forecasts)
# ----------------------------
df["naive_24"]  = df["Price"].shift(24)
df["naive_168"] = df["Price"].shift(168)

# choose a test window (example: last 8760 hours ~ 1 year)
test = df.iloc[-8760:].copy()

# remove rows where baselines not available
test = test.dropna(subset=["naive_24","naive_168","Price"])

m_naive24  = compute_metrics(test["Price"], test["naive_24"])
m_naive168 = compute_metrics(test["Price"], test["naive_168"])

print("Naïve-24:", m_naive24)
print("Naïve-168:", m_naive168)

# ----------------------------
# 4) Plot actual vs naïve (last 200 hours quick view)
# ----------------------------
# plt.figure(figsize=(14,4))
# test["Price"].iloc[-200:].plot(label="Actual")
# test["naive_24"].iloc[-200:].plot(label="Naïve-24")
# test["naive_168"].iloc[-200:].plot(label="Naïve-168")
# plt.title("Actual vs Naïve baselines (last 200 hours)")
# plt.legend()
# plt.tight_layout()
# plt.show()
# (Commented out plot for headless execution, but code is there)
