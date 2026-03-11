import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# LOAD BREAST CANCER DATASET
# -------------------------------------------------
data = load_breast_cancer()
X = data.data          # 30 features
y = data.target        # 0 = malignant, 1 = benign

print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

# -------------------------------------------------
# FEATURE SCALING (VERY IMPORTANT)
# -------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------
# DYNAMIC TEST SIZE (CHANGES EVERY RUN)
# -------------------------------------------------
test_ratio = random.uniform(0.15, 0.35)   # between 15% and 35%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio
)

print("\nDynamic Test Ratio:", round(test_ratio, 2))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -------------------------------------------------
# RANDOM WEIGHTS & BIAS (DYNAMIC)
# -------------------------------------------------
weights = np.random.randn(30) * 0.1   # small weights
bias = np.random.randn() * 0.1

# -------------------------------------------------
# SIGMOID ACTIVATION FUNCTION
# -------------------------------------------------
def sigmoid(net):
    return 1 / (1 + np.exp(-net))

# -------------------------------------------------
# TESTING PHASE (PROBABILITY OUTPUT)
# -------------------------------------------------
net_values = []
sigmoid_outputs = []

print("\n--- Sigmoid Outputs (First 10 Test Samples) ---")
for i in range(10):
    net = np.dot(X_test[i], weights) + bias
    prob = sigmoid(net)

    net_values.append(net)
    sigmoid_outputs.append(prob)

    print(
        f"Sample {i+1}: "
        f"Net = {net:.4f}, "
        f"Sigmoid Output = {prob:.4f}"
    )

# -------------------------------------------------
# GRAPH: NET INPUT vs SIGMOID OUTPUT
# -------------------------------------------------
plt.figure()
plt.scatter(net_values, sigmoid_outputs)
plt.xlabel("Net Input")
plt.ylabel("Sigmoid Output (0 to 1)")
plt.title("Sigmoid Activation Output (Cancer Dataset)")
plt.grid(True)
plt.show()
