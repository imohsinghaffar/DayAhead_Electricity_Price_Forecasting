print("Start debug TF only")
try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
except Exception as e:
    print(f"Error: {e}")
except:
    print("Unknown error")
