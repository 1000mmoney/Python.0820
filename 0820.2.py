# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)
data['Height(Inches)'] = data['Height(Inches)'] * 2.54
data['Weight(Pounds)'] = data['Weight(Pounds)'] * 0.453592

# Extract features and target
X = data[['Height(Inches)']].values  # Features need to be 2D
y = data['Weight(Pounds)'].values    # Target can be 1D

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {MAE:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Values', marker='o')

# Sort X_test for plotting the regression line
sort_index = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[sort_index]
y_pred_sorted = model.predict(X_test_sorted.reshape(-1, 1))

plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Values', linewidth=2, marker='x')

# Graph labels and title
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()