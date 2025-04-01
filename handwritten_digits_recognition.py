import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

path = kagglehub.dataset_download("vbmokin/mnist-models-testing-handwritten-digits")
print("Path to dataset files:", path)

# Extract the data
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units = 25, activation = 'relu', input_shape=(X_train.shape[1],)),
    Dense(units = 10, activation = 'relu'),
    Dense(units = 10, activation = 'softmax'),
    ])

# Compile model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy']
)

# Train model
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_test,y_test))

plt.plot(history.history.get('accuracy'), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy'), label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
