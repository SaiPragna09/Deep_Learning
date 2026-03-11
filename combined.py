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
# RANDOM WEIGHTS & BIAS
# -------------------------------------------------
weights = np.random.randn(30) * 0.1
bias = np.random.randn() * 0.1

# -------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------
def threshold(net):
    return 1 if net >= 0 else 0

def sigmoid(net):
    return 1 / (1 + np.exp(-net))

# -------------------------------------------------
# TESTING PHASE
# -------------------------------------------------
net_values = []
threshold_outputs = []
sigmoid_outputs = []

correct_threshold = 0
correct_sigmoid = 0

print("\n--- Combined Outputs (First 10 Test Samples) ---")
for i in range(len(X_test)):
    net = np.dot(X_test[i], weights) + bias

    t_out = threshold(net)
    s_out = sigmoid(net)
    s_pred = 1 if s_out >= 0.5 else 0

    net_values.append(net)
    threshold_outputs.append(t_out)
    sigmoid_outputs.append(s_out)

    if t_out == y_test[i]:
        correct_threshold += 1
    if s_pred == y_test[i]:
        correct_sigmoid += 1

    if i < 10:
        print(
            f"Sample {i+1}: "
            f"Net={net:.4f}, "
            f"Threshold={t_out}, "
            f"Sigmoid={s_out:.4f}, "
            f"Actual={y_test[i]}"
        )

# -------------------------------------------------
# ACCURACY CALCULATION
# -------------------------------------------------
threshold_accuracy = correct_threshold / len(X_test)
sigmoid_accuracy = correct_sigmoid / len(X_test)

print("\n--- ACCURACY ---")
print("Threshold Accuracy:", threshold_accuracy)
print("Sigmoid Accuracy:", sigmoid_accuracy)

# -------------------------------------------------
# GRAPH 1: THRESHOLD OUTPUT
# -------------------------------------------------
plt.figure()
plt.scatter(net_values[:100], threshold_outputs[:100])
plt.xlabel("Net Input")
plt.ylabel("Threshold Output (0 or 1)")
plt.title("Threshold Activation Output (Cancer Dataset)")
plt.yticks([0, 1])
plt.grid(True)
plt.show()

# -------------------------------------------------
# GRAPH 2: SIGMOID OUTPUT
# -------------------------------------------------
plt.figure()
plt.scatter(net_values[:100], sigmoid_outputs[:100])
plt.xlabel("Net Input")
plt.ylabel("Sigmoid Output (0 to 1)")
plt.title("Sigmoid Activation Output (Cancer Dataset)")
plt.grid(True)
plt.show()
