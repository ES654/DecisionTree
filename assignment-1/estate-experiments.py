
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import rmse, mae
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

max_depth = 5
split = 0.7

estate = pd.read_excel("Real estate valuation data set.xlsx")
shuffled = estate.sample(frac=1).reset_index(drop=True)

X = shuffled.iloc[:, :-1].squeeze()
y = (shuffled.iloc[:, -1:]).T.squeeze()
len_estate = len(y)

X_train, y_train = X.loc[:split*len_estate], y.loc[:split*len_estate]
X_test, y_test = X.loc[split*len_estate+1:].reset_index(
    drop=True), y.loc[split*len_estate+1:].reset_index(drop=True)

print("Please wait for some time, it takes time, you can change max depth if it takes too long time.")

tree = DecisionTree(criterion="information_gain", max_depth=max_depth)
tree.fit(X_train, y_train)
tree.plot()

for depth in range(2, max_depth+1):
    y_hat = tree.predict(X_test, max_depth=depth)
    print("Depth: ", depth)
    print('\tRMSE: ', rmse(y_hat, y_test))
    print('\tMAE: ', mae(y_hat, y_test))

# Decision Tree Regressor from Sci-kit learn
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)
y_hat = pd.Series(dt.predict(X_test))

print('Sklearn RMSE: ', rmse(y_hat, y_test))
print('Sklearn MAE: ', mae(y_hat, y_test))
