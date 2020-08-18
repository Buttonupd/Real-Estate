import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import xlrd
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

my_data = pd.read_excel(r'C:\Users\User\Downloads\ENB2012_data.xlsx')
my_data.replace([np.inf, -np.inf], np.nan, inplace=True)
my_data.fillna(99, inplace=True)
my_data.astype(np.float64)

my_data = my_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']]
predict = "X8"
X = np.nan_to_num(my_data.drop([predict], 1))
y = np.nan_to_num(my_data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=20, train_size=80)

best = 0

for _ in range(130):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=20, train_size=80)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy

    with open('ENB.pickle', 'wb') as f:
        pickle.dump(linear, f)

pickle_in = open("ENB.pickle", "rb")
linear = pickle.load(pickle_in)

"""Retrieve coefficients from y= mx+b"""

print("Coefficient \n", linear.coef_)
print("Intercept\n ", linear.intercept_)

"""Visualize predictions"""
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "X2"
style.use("ggplot")
plt.scatter(my_data[p], my_data["X8"])

plt.xlabel(p)
plt.ylabel('X8')
plt.show()

