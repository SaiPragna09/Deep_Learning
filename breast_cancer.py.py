# Decision Tree - Breast Cancer Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load dataset
cancer = load_breast_cancer()

# Convert to DataFrame
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data['Target'] = cancer.target

# Map target values
data['Target'] = data['Target'].map({0: 'Malignant', 1: 'Benign'})

# Basic info
print("\nShape:", data.shape)
print("\nFirst rows:\n", data.head())
print("\nClass distribution:\n", data['Target'].value_counts())

# Features and target
X = data.drop("Target", axis=1)
y = data["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# Train model
model = DecisionTreeClassifier(criterion="gini", max_depth=4)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Feature importance
importance = model.feature_importances_
features = X.columns

print("\nFeature Importance:")
for f, i in zip(features, importance):
    print(f, ":", round(i, 3))

# Plot feature importance (top 10)
plt.figure()
sorted_idx = importance.argsort()[-10:]
plt.barh(features[sorted_idx], importance[sorted_idx])
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")

# Plot decision tree (limited depth for clarity)
plt.figure(figsize=(12,8))
tree.plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    max_depth=3
)
plt.title("Decision Tree (Cancer Dataset)")
plt.savefig("decision_tree.png")