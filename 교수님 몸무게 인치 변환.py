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

array = data.values
X = array[:, 0]
Y = array[:, 1]
X = X.reshape(-1, 1) # 2차원 행열일 떈 필요 없다 1차원 인경우 [ ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_predition = model.predict((X_test))
mae = mean_absolute_error(Y_predition, Y_test)
print(mae)

plt.scatter(X_test, Y_predition, color='red')
plt.scatter(X_test, Y_test, color='blue')
plt.xlabel("Height(cm)")
plt.ylabel("Weghit(kg)")
plt.show()