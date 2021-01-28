"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import information_gain, gini_gain, regression_impurity

np.random.seed(42)


class treenode():
    def __init__(self, split_col=None, value=None, depth=None):
        self.value = value          # Leaf Value
        self.split_col = split_col  # Attribute to bec checked
        self.children = {}          # child nodes
        self.prob = None            # For Attributes not present
        self.mean = None            # For Real Attribute
        self.depth = depth

    def print_node(self, indent=0):
        lookup = {"low": "<", "high": ">"}
        if self.split_col != None:
            for child in self.children:
                if self.children[child].prob != None:
                    print("  "*indent, "|?(X({}) = {}):".format(self.split_col, child))
                else:
                    print("  "*indent, "|?(X({}) {} {:.2f}):".format(self.split_col,
                                                                     lookup[child], self.mean))
                self.children[child].print_node(indent+1)
        else:
            if type(self.value) == str:
                print("  "*indent,
                      "|Value = {} Depth = {}".format(self.value, self.depth))
            else:
                print(
                    "  "*indent, "|Value = {:.2f} Depth = {}".format(self.value, self.depth))

    def getVal(self, X, max_depth=np.inf):
        if self.split_col == None or self.depth >= max_depth:
            return self.value
        else:
            if self.mean == None:
                if X[self.split_col] in self.children:
                    # When feature is already seen
                    return self.children[X[self.split_col]].getVal(X.drop(self.split_col), max_depth=max_depth)
                else:
                    # When Feature is not in train class
                    max_prob = 0
                    child_name = None
                    for child in self.children:
                        if child.prob > max_prob:
                            max_prob = child.prob
                            child_name = child
                    return child.getVal(X.drop(self.split_col), max_depth=max_depth)
            else:
                if X[self.split_col] <= self.mean:
                    return self.children["low"].getVal(X, max_depth=max_depth)
                else:
                    return self.children["high"].getVal(X, max_depth=max_depth)


class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=100):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        # criterion won't be used for regression
        > criterion : {"information_gain", "gini_index"}
        > max_depth : The maximum depth the tree can grow to
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.Ydtype = None
        self.colname = None

    def create_tree(self, X, Y, parent_node, split_col=None, depth=0):
        if Y.unique().size == 1:
            return treenode(value=Y.values[0], depth=depth)

        if len(X.columns) <= 0 or depth >= self.max_depth:
            if str(Y.dtype) == 'category':
                return treenode(value=Y.mode(dropna=True)[0], depth=depth)
            else:
                return treenode(value=Y.mean(), depth=depth)

        max_inf_gain = -np.inf
        max_mean = None

        for column in list(X.columns):
            mean_val = None

            if str(Y.dtype) == "category":
                # for classification
                if self.criterion == "information_gain":
                    col_inf_gain = information_gain(Y, X[column])
                elif self.criterion == "gini_index":
                    col_inf_gain = gini_gain(Y, X[column])
            else:
                # for regression
                col_inf_gain = regression_impurity(Y, X[column])

            if type(col_inf_gain) == tuple:
                # If attribute selected is range of values
                mean_val = col_inf_gain[1]
                col_inf_gain = col_inf_gain[0]

            if col_inf_gain > max_inf_gain:
                max_inf_gain = col_inf_gain
                split_col = column
                max_mean = mean_val

        node = treenode(split_col=split_col)
        parent_col = X[split_col]
        if str(parent_col.dtype) == "category":
            X = X.drop(split_col, axis=1)
            split_col_classes = parent_col.groupby(parent_col).count()

            for cat in list(split_col_classes.index):
                sub_rows = parent_col == cat
                if sub_rows.sum() > 0:
                    node.children[cat] = self.create_tree(
                        X[sub_rows],
                        Y[sub_rows],
                        node,
                        depth=depth+1)
                    node.children[cat].prob = len(X[sub_rows])/len(X)
                    # node.children[cat].prob = X[sub_rows].size/X.size
        else:
            low_index = parent_col < max_mean
            high_index = parent_col > max_mean
            node.children["low"] = self.create_tree(
                X[low_index],
                Y[low_index],
                node,
                depth=depth+1)

            node.children["high"] = self.create_tree(
                X[high_index],
                Y[high_index],
                node,
                depth=depth+1)
            node.mean = max_mean
        if str(Y.dtype) == 'category':
            node.value = Y.mode(dropna=True)[0]
        else:
            node.value = Y.mean()
        node.depth = depth
        return node

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.Ydtype = y.dtype
        self.colname = y.name
        self.root = self.create_tree(X, y, None)

        pass

    def predict(self, X, max_depth=np.inf):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Y = []
        for x in X.index:
            Y.append(self.root.getVal(X.loc[x], max_depth=max_depth))
        return pd.Series(Y, name=self.colname).astype(self.Ydtype)
        # pass

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.root.print_node(indent=0)
        pass
