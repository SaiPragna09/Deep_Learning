import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

w = 0.0
b = 0.0

lr = 0.01
epochs = 50

beta = 0.9

vw = 0
vb = 0

for epoch in range(epochs):

    y_pred = w * X + b

    dw = -2 * np.mean(X * (Y - y_pred))
    db = -2 * np.mean(Y - y_pred)

    vw = beta * vw + (1 - beta) * dw
    vb = beta * vb + (1 - beta) * db

    w = w - lr * vw
    b = b - lr * vb

print("Weight:", w)
print("Bias:", b)