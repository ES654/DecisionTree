import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

split = 0.8         # Split Ratio

# Loading data
iris = pd.read_csv("iris.csv")
iris["variety"] = iris["variety"].astype("category")
shuffled = iris.sample(frac=1).reset_index(drop=True)

# Preprocessing data
X = shuffled.iloc[:, :-1].squeeze()
X = X[['sepal.width', "petal.width"]]
y = (shuffled.iloc[:, -1:]).T.squeeze()
len_iris = len(y)

# Splitting Data
X_train, y_train = X.loc[:split*len_iris], y.loc[:split*len_iris]
X_test, y_test = X.loc[split*len_iris+1:].reset_index(
    drop=True), y.loc[split*len_iris+1:].reset_index(drop=True)

# Learning classifier
tree = RandomForestClassifier(10, criterion="information_gain", max_attr=2)
tree.fit(X_train, y_train)

# Predicition
y_hat = tree.predict(X_test)
print("Please wait, preparing plots...", flush=True)
tree.plot(X, y, db=True)

# Calculating Metrics
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
