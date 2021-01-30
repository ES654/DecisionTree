from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
from sklearn import tree as sktree


class AdaBoostClassifier():
    # Optional Arguments: Type of estimator
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=5,
                 max_depth=1, criterion="entropy"):
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.alphas = []

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        weights = np.ones(len(y))/len(y)
        for n in tqdm(range(self.n_estimators)):
            # Learning and Predicting
            tree = self.base_estimator(
                criterion=self.criterion, max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=weights)
            y_hat = pd.Series(tree.predict(X))

            # Calculating error and alpha
            mis_idx = y_hat != y
            err_m = np.sum(weights[mis_idx])/np.sum(weights)
            alpha_m = 0.5*np.log((1-err_m)/err_m)

            # Updating Weights
            weights[mis_idx] *= np.exp(alpha_m)
            weights[~mis_idx] *= np.exp(-alpha_m)
            # Normalizing weights
            weights /= np.sum(weights)

            # Storing trees
            self.trees.append(tree)
            self.alphas.append(alpha_m)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        final = None

        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.trees)):
            if final is None:
                final = pd.Series(tree.predict(X))
                final[0] *= alpha_m
            else:
                final += pd.Series(tree.predict(X))*alpha_m
        return final.apply(np.sign)

    def plot(self, X, y, name=""):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        assert(len(list(X.columns)) == 2)
        color = ["r", "b", "g"]
        # ax1 = plt.figure(figsize=(len(self.trees)*5, 4))
        Zs = None
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.trees)):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            print(sktree.export_text(tree))
            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            # _ = ax1.add_subplot(1, len(self.trees), i + 1)
            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if Zs is None:
                Zs = alpha_m*Z
            else:
                Zs += alpha_m*Z
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)
            for y_label in y.unique():
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],
                               cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        com_surface = np.sign(Zs)
        cs = ax2.contourf(xx, yy, com_surface, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")

        # Saving Figures
        fig1.savefig("Q5_{}Fig1.png".format(name))
        fig2.savefig("Q5_{}Fig2.png".format(name))
        return fig1, fig2
