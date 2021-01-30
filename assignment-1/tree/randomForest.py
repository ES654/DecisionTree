from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode


class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=10, max_attr=3):
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
            X_sub = X.sample(n=np.random.randint(
                1, self.max_attr + 1), axis='columns')
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
            if y_hat_total is None:
                y_hat_total = tree.predict(X).to_frame()
            else:
                y_hat_total[i] = tree.predict(X)
        return y_hat_total.mode(axis=1)[0]

    def plot(self, X, y, db=False):
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
        if db:
            return self.decisionBoundary(X, y)

    def decisionBoundary(self, X, y):
        assert(len(list(X.columns)) == 2)

        color = ["r", "y", "b"]
        lookup = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min
        Zs = []

        for i, tree in enumerate(self.trees):
            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))
            Z = tree.predict(pd.DataFrame(
                np.c_[xx.ravel(), yy.ravel()], columns=list(X.columns))).to_numpy()
            Z = np.vectorize(lambda x: lookup[x])(Z)
            Z = Z.reshape(xx.shape)
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Zs.append(Z)
            for y_label in y.unique():
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],
                               cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Zs = np.array(Zs)
        com_surface, _ = weighted_mode(Zs, np.ones(Zs.shape))
        Z = np.mean(Zs, axis=0)
        cs = ax2.contourf(xx, yy, com_surface[0], cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend()
        ax2.set_title("Common Decision Surface")

        # Saving Figures
        fig1.savefig("Q7_Fig1.png")
        fig2.savefig("Q7_Fig2.png")
        return fig1, fig2


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=10, max_attr=3):
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
