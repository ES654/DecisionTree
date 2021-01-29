from .base import DecisionTree
import numpy as np
import pandas as pd
from tqdm import tqdm


class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=100, max_attr=3):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.max_attr = max_attr

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for n in tqdm(range(self.n_estimators)):
            X_sub = X.sample(n=self.max_attr, axis='columns')
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth)
            tree.fit(X_sub, y)
            self.trees.append(tree)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat_total = None
        for i, tree in enumerate(self.trees):
            if y_hat_total is No3ne:
                y_hat_total = tree.predict(X).to_frame()
            else:
                y_hat_total[i] = tree.predict(X)
        return y_hat_total.mode(axis=1)[0]

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for i, tree in enumerate(self.trees):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            tree.plot()


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=100, max_attr=3):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.max_attr = max_attr

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for n in tqdm(range(self.n_estimators)):
            X_sub = X.sample(n=self.max_attr, axis='columns')
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth)
            tree.fit(X_sub, y)
            self.trees.append(tree)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat_total = None
        for i, tree in enumerate(self.trees):
            if y_hat_total is None:
                y_hat_total = tree.predict(X).to_frame()
            else:
                y_hat_total[i] = tree.predict(X)
        return y_hat_total.mean(axis=1)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i, tree in enumerate(self.trees):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            tree.plot()
