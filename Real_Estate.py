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

my_data = pd.read_excel(r'C:\Users\User\Daniel Mwihaki\PycharmProjects\Automation\Real estate valuation data sett.xlsx')


# print(pd.isnull(my_data).sum())


# def clean_data_set(my_data):
#     data = my_data
#     print(data)
#     assert isinstance(my_data, pd.DataFrame)
#     my_data.dropna(inplace=True)
#     indices_to_keep = ~my_data.isin([np.nan, np.inf, -np.inf]).any(1)
#     print(indices_to_keep)
#     return my_data[indices_to_keep].astype(np.float64)


# my_data.replace([np.inf, -np.inf], np.nan, inplace=True)
# my_data.fillna(99, inplace=True)
# my_data.astype(np.float64)
# my_data = my_data.reset_index()
# print(my_data)

"""Categorize using Data_frame"""

my_data = my_data[["No", "X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
                   "X4 number of convenience stores", "X5 latitude", "X6 longitude", "Y house price of unit area"]]

predict = "Y house price of unit area"

X = np.array(my_data.drop([predict], 1))
y = np.array(my_data[predict])

# X = np.nan_to_num(my_data.drop([predict], 1))
# y = np.nan_to_num(my_data[predict])


"""Train and save Data"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=20, train_size=80)

best = 0
for _ in range(130):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=20, train_size=80)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(acc)
    if acc > best:
        best = acc
    with open("model.pickle", "wb") as f:
        pickle.dump(linear, f)

"""Open saved data or model"""

pickle_in = open("model.pickle", "rb")
linear = pickle.load(pickle_in)

"""Retrieve coefficients from y= mx+b"""

print("Coefficient \n", linear.coef_)
print("Intercept\n ", linear.intercept_)

"""Visualize predictions"""
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "X2 house age"
style.use("ggplot")
plt.scatter(my_data[p], my_data["Y house price of unit area"])

plt.xlabel(p)
plt.ylabel('Y house price of unit area')
plt.show()
