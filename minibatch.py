import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

w = 0.0
b = 0.0

lr = 0.01
epochs = 50
batch_size = 2

for epoch in range(epochs):

    for i in range(0, len(X), batch_size):

        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        y_pred = w * X_batch + b

        dw = -2 * np.mean(X_batch * (Y_batch - y_pred))
        db = -2 * np.mean(Y_batch - y_pred)

        w = w - lr * dw
        b = b - lr * db

print("Weight:", w)
print("Bias:", b)