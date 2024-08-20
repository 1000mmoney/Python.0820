# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess data
culums = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=culums)

# Check the first few rows of the data


array = data.values
X = array[:, :-1]  # Features
Y = array[:, -1]   # Target

# Min-Max Scaling
scaler = MinMaxScaler()
rescaled_X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.3, random_state=42)

# Model selection and training
model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, Y_train)

# Predict on test data
Y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# K-Fold cross-validation
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='neg_mean_squared_error')
print(f"Mean Cross-Validation MSE: {-results.mean():.4f}")

# Plotting the predicted values vs. actual values
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, edgecolor='k', color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', linewidth=2)

plt.show()
