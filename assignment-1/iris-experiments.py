import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import accuracy, precision, recall
from tqdm import tqdm

np.random.seed(42)

split = 0.7

iris = pd.read_csv("iris.csv")
iris["variety"] = iris["variety"].astype("category")

shuffled = iris.sample(frac=1).reset_index(drop=True)

X = shuffled.iloc[:, :-1].squeeze()
y = (shuffled.iloc[:, -1:]).T.squeeze()
len_iris = len(y)

X_train, y_train = X.loc[:split*len_iris], y.loc[:split*len_iris]
X_test, y_test = X.loc[split*len_iris+1:].reset_index(
    drop=True), y.loc[split*len_iris+1:].reset_index(drop=True)


tree = DecisionTree(criterion="information_gain")
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
tree.plot()

print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))


# Optimizing for depth
def optimize_depth(X, y, folds=5, depths=[3, 4, 5, 6]):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    max_depth = max(depths)
    trees = {}
    accuracies = {}
    chunk = int(len(X)//folds)

    for fold in tqdm(range(folds)):
        indices = range(fold*chunk, (fold+1)*chunk)
        curr_fold = pd.Series([False for i in range(len(X))])
        curr_fold.loc[indices] = True

        X_train, y_train = X[~curr_fold].reset_index(
            drop=True), y[~curr_fold].reset_index(drop=True)
        X_test, y_test = X[curr_fold].reset_index(
            drop=True), y[curr_fold].reset_index(drop=True)

        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)
        trees[fold+1] = tree

        for depth in depths:
            y_hat = tree.predict(X_test, max_depth=depth)
            if fold+1 in accuracies:
                accuracies[fold+1][depth] = accuracy(y_hat, y_test)
            else:
                accuracies[fold+1] = {depth: accuracy(y_hat, y_test)}

    accuracies = pd.DataFrame(accuracies).transpose()
    accuracies.index.name = "Fold Number"
    accuracies.loc["mean"] = accuracies.mean()
    print(accuracies)
    print("Best Mean Accuracy = {}".format(accuracies.loc["mean"].max()))
    print("Optimum Depth = {}".format(accuracies.loc["mean"].idxmax()))


optimize_depth(X, y, 4, [2, 3, 4, 5, 6, 7, 8, 10])
