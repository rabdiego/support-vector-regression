# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scalling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Training the model
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train.reshape(-1)) 

# Evaluating the model
X_range = np.arange(min(sc_X.inverse_transform(X_test)), max(sc_X.inverse_transform(X_test)), 0.1)
X_range = X_range.reshape(-1, 1)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_range)).reshape(-1, 1))

# Plotting the results
plt.title('Position x Salary', c='m')
plt.xlabel('Position', c='m')
plt.ylabel('Salary', c='m')
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test), c='m')
plt.plot(X_range, y_pred, c='c')
plt.legend(['Real values', 'Predicted values'])
plt.show()
