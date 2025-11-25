# train_mnist.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Make sure models/ directory exists
os.makedirs("models", exist_ok=True)

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # (N, 28, 28, 1)
x_test  = x_test[..., tf.newaxis]

# 2. Define a simple CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 3. Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 4. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 5. Save as TensorFlow SavedModel
save_path = "models/mnist_cnn_saved_model"
model.save(save_path)
print(f"Model saved to {save_path}")
