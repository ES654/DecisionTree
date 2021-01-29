"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 5
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
y = y.cat.rename_categories([-1, 1])  # Changing 0 to -1 for adaboost


criteria = 'entropy'
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(
    base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X, y)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


# AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
