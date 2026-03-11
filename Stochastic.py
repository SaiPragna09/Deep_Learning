import numpy as np

# Data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# Initialize parameters
w = 0.0
b = 0.0

lr = 0.01
epochs = 50

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        y_pred = w * X[i] + b

        dw = -2 * X[i] * (Y[i] - y_pred)
        db = -2 * (Y[i] - y_pred)

        w = w - lr * dw
        b = b - lr * db

print("Weight:", w)
print("Bias:", b)