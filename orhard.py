import numpy as np
import matplotlib.pyplot as plt

# ---------------- INPUT AND OUTPUT ----------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [1]])

print("OR GATE SINGLE PERCEPTRON - HARD-CODED WEIGHTS")
print("=" * 60)

# ---------------- HARD-CODED WEIGHTS & BIAS ----------------
weights = np.array([[1.0],
                    [1.0]])
bias = np.array([[-0.5]])

print(f"Weights: w1={weights[0,0]:.1f}, w2={weights[1,0]:.1f}")
print(f"Bias: b={bias[0,0]:.1f}")
print(f"Equation: {weights[0,0]:.1f}x + {weights[1,0]:.1f}y + {bias[0,0]:.1f} > 0")

# ---------------- STEP ACTIVATION FUNCTION ----------------
def step_function(x):
    return 1 if x > 0 else 0

# ---------------- PERCEPTRON PREDICTION ----------------
def perceptron_predict(X, weights, bias):
    linear = np.dot(X, weights) + bias
    predictions = np.zeros((X.shape[0], 1))
    for i in range(len(linear)):
        predictions[i, 0] = step_function(linear[i, 0])
    return predictions

# ---------------- PREDICTIONS ----------------
print("\nPREDICTIONS")
linear_values = np.dot(X, weights) + bias
predictions = perceptron_predict(X, weights, bias)

for i in range(len(X)):
    print(f"[{X[i,0]}, {X[i,1]}] -> Linear: {linear_values[i,0]:.1f} | "
          f"Pred: {int(predictions[i,0])} | Actual: {int(y[i,0])}")

# ---------------- DECISION BOUNDARY GRAPH ----------------
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1],
            s=600, c='red', marker='o', edgecolors='black',
            linewidth=4, label='Class 0', zorder=3)

plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1],
            s=600, c='blue', marker='s', edgecolors='black',
            linewidth=4, label='Class 1', zorder=4)

x1_range = np.linspace(-0.3, 1.3, 100)
x2_boundary = (-1 * x1_range + 0.5)

plt.plot(x1_range, x2_boundary, 'g-', linewidth=5,
         label='Decision Boundary', zorder=2)

plt.fill_between(x1_range, x2_boundary, 1.3, alpha=0.2, color='blue')
plt.fill_between(x1_range, -0.3, x2_boundary, alpha=0.2, color='red')

plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.xlabel("Input 1 (x)")
plt.ylabel("Input 2 (y)")
plt.title("Decision Boundary\nx + y - 0.5 = 0")
plt.legend()
plt.grid(True, alpha=0.3)

# ---------------- LINEAR COMBINATION BAR CHART ----------------
plt.subplot(1, 2, 2)
samples = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
linear_vals = linear_values.flatten()

plt.bar(samples, linear_vals,
        color=['red', 'blue', 'blue', 'blue'],
        edgecolor='black', linewidth=2)

plt.axhline(y=0, color='black', linestyle='--',
            linewidth=2, label='Threshold')
plt.ylabel("w1*x1 + w2*x2 + b")
plt.title("Linear Combination Values")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------- VERIFICATION ----------------
print("\nVERIFICATION RESULTS:")
print("Decision boundary equation: x + y - 0.5 = 0")

for i in range(len(X)):
    z = weights[0,0]*X[i,0] + weights[1,0]*X[i,1] + bias[0,0]
    print(f"[{X[i,0]}, {X[i,1]}] -> {z:.3f} -> "
          f"{'Class 1' if z > 0 else 'Class 0'}")

# ---------------- POSITION RELATIVE TO BOUNDARY ----------------
print("\nPoint positions relative to boundary:")
for i in range(len(X)):
    value = weights[0,0]*X[i,0] + weights[1,0]*X[i,1] + bias[0,0]
    side = "Above (Class 1)" if value > 0 else "Below (Class 0)"
    print(f"[{X[i,0]}, {X[i,1]}]: {value:.3f} -> {side}")
