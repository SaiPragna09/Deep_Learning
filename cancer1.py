import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATASET
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# -----------------------------
# INITIALIZE WEIGHTS & BIAS
# -----------------------------
weights = np.random.rand(30)
bias = np.random.rand()

# -----------------------------
# THRESHOLD ACTIVATION
# -----------------------------
def threshold(net):
    return 1 if net >= 0 else 0

# -----------------------------
# TESTING PHASE
# -----------------------------
correct = 0
net_values = []
outputs = []

print("\n--- Threshold Neuron Output (First 5 Test Samples) ---")
for i in range(len(X_test)):
    net = np.dot(X_test[i], weights) + bias
    out = threshold(net)

    net_values.append(net)
    outputs.append(out)

    if out == y_test[i]:
        correct += 1

    if i < 5:
        print(f"Sample {i+1}: Net={net:.2f}, Predicted={out}, Actual={y_test[i]}")

accuracy = correct / len(X_test)
print("\nThreshold Accuracy:", accuracy)

# -----------------------------
# GRAPH
# -----------------------------
plt.figure()
plt.scatter(net_values[:50], outputs[:50])
plt.xlabel("Net Input")
plt.ylabel("Threshold Output")
plt.title("Threshold Activation Function Output")
plt.show()
