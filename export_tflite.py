# export_tflite.py
import tensorflow as tf
import os

saved_model_dir = "models/mnist_cnn_saved_model"
os.makedirs("models", exist_ok=True)

# Basic float32 TFLite model (simplest for Raspberry Pi)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

tflite_path = "models/mnist_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_path}")
