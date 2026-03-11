import numpy as np
import matplotlib.pyplot as plt

# ---------------- Activation Functions ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------- XOR Dataset ----------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ---------------- Network Parameters ----------------
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

# ---------------- Random Initialization ----------------
np.random.seed(42)

W_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
b_hidden = np.random.uniform(-1, 1, (1, hidden_size))

W_output = np.random.uniform(-1, 1, (hidden_size, output_size))
b_output = np.random.uniform(-1, 1, (1, output_size))

# ---------------- Training ----------------
loss_history = []
print("Training XOR Neural Network with Random Initialization...")

for i in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W_output) + b_output
    predicted_output = sigmoid(final_input)

    # Error and loss
    error = y - predicted_output
    loss = np.mean(np.abs(error))
    loss_history.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    d_hidden = d_output.dot(W_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if i % 2000 == 0:
        print(f"Epoch {i}, Loss = {loss:.4f}")

# ---------------- Final Output ----------------
print("\nXOR Gate Output (Rounded):")
for i in range(len(X)):
    print(X[i], "->", round(predicted_output[i][0]))

# ---------------- Loss Graph ----------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve for XOR Neural Network")
plt.show()

# ---------------- Decision Boundary ----------------
h = 0.01
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
hidden_layer = sigmoid(np.dot(grid, W_hidden) + b_hidden)
output_layer = sigmoid(np.dot(hidden_layer, W_output) + b_output)
Z = output_layer.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.85)
plt.contour(xx, yy, Z, levels=[0.5], colors='yellow', linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(),
            cmap=plt.cm.coolwarm, edgecolors='k', s=100)
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("Decision Boundary for XOR Gate")
plt.show()

# ---------------- BAR CHART: LEARNED WEIGHTS ----------------
plt.figure(figsize=(8, 4))
weights = np.concatenate((W_hidden.flatten(), W_output.flatten()))
labels = [f"W{i+1}" for i in range(len(weights))]
plt.bar(labels, weights)
plt.xlabel("Weights")
plt.ylabel("Value")
plt.title("Learned Weights after Training")
plt.grid(axis='y', alpha=0.3)
plt.show()

# ---------------- BAR CHART: LEARNED BIASES ----------------
plt.figure(figsize=(6, 4))
biases = np.concatenate((b_hidden.flatten(), b_output.flatten()))
labels = [f"B{i+1}" for i in range(len(biases))]
plt.bar(labels, biases, color='orange')
plt.xlabel("Biases")
plt.ylabel("Value")
plt.title("Learned Biases after Training")
plt.grid(axis='y', alpha=0.3)
plt.show()
