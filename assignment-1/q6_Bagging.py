"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree

from sklearn.tree import DecisionTreeClassifier
# Or use sklearn decision tree
# from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################

N = 50
P = 2
NUM_OP_CLASSES = 2
n_estimators = 5
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = 'information_gain'
tree = DecisionTreeClassifier
Classifier_B = BaggingClassifier(
    base_estimator=tree, n_estimators=n_estimators,
    criterion="gini")
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot(X, y)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

print("Plots saved as Q6_Fig1.png and Q6_Fig2.png")
