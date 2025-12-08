import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("Starting imports...")

try:
    print("Importing pandas...")
    import pandas as pd
    print("Pandas imported.")
except Exception as e:
    print(f"Pandas failed: {e}")

try:
    print("Importing numpy...")
    import numpy as np
    print("Numpy imported.")
except Exception as e:
    print(f"Numpy failed: {e}")

try:
    print("Importing matplotlib...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Matplotlib imported.")
except Exception as e:
    print(f"Matplotlib failed: {e}")

try:
    print("Importing xgboost...")
    import xgboost as xgb
    print("XGBoost imported.")
except Exception as e:
    print(f"XGBoost failed: {e}")

try:
    print("Importing tensorflow...")
    import tensorflow as tf
    print("TensorFlow imported.")
except Exception as e:
    print(f"TensorFlow failed: {e}")

print("All imports finished.")
