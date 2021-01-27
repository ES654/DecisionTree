import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

iris = pd.read_csv("assignment-1\iris.csv")
iris["variety"] = iris["variety"].astype("category")

shuffled = iris.sample(frac=1)

X = shuffled.iloc[:, :-1].squeeze()
y = (shuffled.iloc[:, -1:]).T.squeeze()

len_iris = len(y)


X_train, y_train = X.loc[:0.7*len_iris], y.loc[:0.7*len_iris]
X_test, y_test = X.loc[0.7*len_iris+1:].reset_index(
    drop=True), y.loc[0.7*len_iris+1:].reset_index(drop=True)


tree = DecisionTree(criterion="information_gain")
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
tree.plot()

print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

# 5-fold cross-validation
