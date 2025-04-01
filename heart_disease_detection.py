import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Fetch dataset 
heart_disease = fetch_ucirepo(id=45)

# Data (as pandas dataframes) 
X = heart_disease.data.features
y = heart_disease.data.targets

X = X.dropna()
y = y.loc[X.index]

print(len(X))

# Split data into training and test sets
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)

# Build model
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=5, activation='softmax')  # Assuming 4 classes
])

# Compile model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_cv,y_cv))

# Plot training & validation accuracy

plt.plot(history.history.get('accuracy'), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy'), label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Evaluate model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
