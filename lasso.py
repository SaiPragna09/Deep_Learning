import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lasso model
lasso = Lasso(alpha=0.1)

# Train
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

print("Simple Lasso Regression")

print("MSE:", mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

print("Coefficients (first 5):", lasso.coef_[:5])

print("Intercept:", lasso.intercept_)