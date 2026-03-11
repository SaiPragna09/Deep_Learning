import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------------------------
# LOAD BREAST CANCER DATASET
# -------------------------------------------------
data = load_breast_cancer()
X = data.data          # 30 features
y = data.target        # 0 or 1

print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------
# DYNAMIC TRAIN-TEST SPLIT
# -------------------------------------------------
test_ratio = random.uniform(0.15, 0.35)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio
)

print("\nDynamic Test Ratio:", round(test_ratio, 2))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -------------------------------------------------
# NEURAL NETWORK WITH ONE HIDDEN LAYER (12 NEURONS)
# -------------------------------------------------
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(30,)))
model.add(Dense(1, activation='sigmoid'))

# -------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# TRAINING PHASE (ACTUAL LEARNING)
# -------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=16,
    verbose=1
)

# -------------------------------------------------
# TESTING PHASE
# -------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n--- TEST RESULTS ---")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# -------------------------------------------------
# GRAPH: TRAINING ACCURACY
# -------------------------------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epochs")
plt.grid(True)
plt.show()
