"""Dependencies"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import xlrd
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

"""Load Data"""

my_data = pd.read_excel(r'C:\Users\User\Daniel Mwihaki\Downloads\Real estate valuation data set.xlsx')

print(my_data.keys())

"""Categorize using Data_frame"""

my_data = my_data[["No", "X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
                   "X4 number of convenience stores", "X5 latitude", "X6 longitude", "Y house price of unit area"]]

predict = "Y house price of unit area"

X = np.array(my_data.drop([predict], 1))
y = np.array(my_data[predict])

# X = np.nan_to_num(my_data.drop([predict]))
# y = np.nan_to_num(my_data[predict])
# #
# X = np.isnan(my_data.drop([predict], 1))
#
# y = np.isnan(my_data[predict])
"""Train and save Data"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8)

best = 0
for _ in range(130):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train.squeeze())

    acc = linear.score(x_test, y_test)

    print(acc)
    if acc > best:
        best = acc
    with open("Valuation.pickle", "wb") as f:
        pickle.dump(linear, f)

"""Open saved data or model"""

pickle_in = open("Valuation.pickle", "rb")
linear = pickle.load(pickle_in)

"""Retrieve coefficients from y= mx+b"""

print("Coefficient \n", linear.coef_)
print("Intercept\n ", linear.intercept_)

"""Visualize predictions"""
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


p = "X3 distance to the nearest MRT station"
style.use("ggplot")
plt.scatter(my_data[p], my_data["Y house price of unit area"])

plt.xlabel(p)
plt.ylabel("Financial Output")
plt.show()

