print("Start debug")
try:
    import pandas as pd
    print("Pandas imported")
    import numpy as np
    print("Numpy imported")
    import xgboost
    print("XGBoost imported")
    import tensorflow as tf
    print("TensorFlow imported")
    from tensorflow import keras
    print("Keras imported")
    from LSTM.lstm_model import LSTMModel
    print("LSTMModel imported")
except Exception as e:
    print(f"Error: {e}")
print("End debug")
