# matplotlib 설치
import matplotlib.pyplot as plt

#pandas 설치
import pandas as pd

# numpy 설치
import numpy as np

# scikit-learn 설치
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load and preprocess data
culums = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=culums)

array = data.values
X = array[:, 0:13]
Y = array[:, 13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue')
# plt.scatter(range(len(X_test[:15])), Y_pred[:15], color='red', marker='*')
# plt.xlabel("index")
# plt.ylabel("MEDV($1,000)")


kfold = KFold(n_splits=5)
mse = cross_val_score(model, X, Y, scoring='neg_mean_squared_error')
print(model.coef_, model.intercept_)
print(mse.mean())

