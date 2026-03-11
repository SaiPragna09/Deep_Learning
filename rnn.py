import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# -------------------------------
# Activation Functions
# -------------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-12))

def clip_gradients(grad, clip_value):
    return np.clip(grad, -clip_value, clip_value)


# -------------------------------
# Initialize Weights
# -------------------------------
Wxh = np.array([[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]])

Whh = np.array([[0.7, 0.8],
                [0.9, 1.0]])

Why = np.array([[0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6]])

bh = np.array([[0.1],
               [0.2]])

by = np.array([[0.1],
               [0.2],
               [0.3]])


# -------------------------------
# Input Sequence
# -------------------------------
x1 = np.array([[1],[0],[0]])
x2 = np.array([[0],[1],[0]])
x3 = np.array([[0],[0],[1]])

inputs = [x1, x2, x3]


# -------------------------------
# Target Sequence
# -------------------------------
t1 = np.array([[0],[1],[0]])
t2 = np.array([[0],[0],[1]])
t3 = np.array([[1],[0],[0]])

targets = [t1, t2, t3]


# -------------------------------
# Training Parameters
# -------------------------------
learning_rate = 0.1
threshold = 0.05
max_epochs = 1000
clip_value = 5

loss_history = []


# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(max_epochs):

    h0 = np.zeros((2,1))
    hs = [h0]
    ps = []

    loss = 0


    # -------- Forward Pass --------
    for t in range(3):

        ht = np.tanh(Wxh @ inputs[t] + Whh @ hs[t] + bh)

        y = Why @ ht + by
        p = softmax(y)

        hs.append(ht)
        ps.append(p)

        loss += cross_entropy(targets[t], p)


    loss_history.append(loss)


    if loss < threshold:
        print("Training converged at epoch:", epoch+1)
        break


    # -------- Initialize Gradients --------
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)

    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    dh_next = np.zeros((2,1))


    # -------- Backpropagation Through Time --------
    for t in reversed(range(3)):

        dy = ps[t] - targets[t]

        dWhy += dy @ hs[t+1].T
        dby += dy

        dh = Why.T @ dy + dh_next

        dh_raw = (1 - hs[t+1]**2) * dh

        dbh += dh_raw
        dWxh += dh_raw @ inputs[t].T
        dWhh += dh_raw @ hs[t].T

        dh_next = Whh.T @ dh_raw


    # -------- Gradient Clipping --------
    dWxh = clip_gradients(dWxh, clip_value)
    dWhh = clip_gradients(dWhh, clip_value)
    dWhy = clip_gradients(dWhy, clip_value)

    dbh = clip_gradients(dbh, clip_value)
    dby = clip_gradients(dby, clip_value)


    # -------- Update Weights --------
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy

    bh -= learning_rate * dbh
    by -= learning_rate * dby


    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss)


# -------------------------------
# Final Results
# -------------------------------
print("\nFinal Loss:", loss)

print("\nFinal Predictions:")
for t in range(3):
    print(np.round(ps[t], 2))


# -------------------------------
# Plot Loss Curve
# -------------------------------
plt.plot(loss_history, color='blue')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()