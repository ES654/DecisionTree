import numpy as np


def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    assert(Y.size > 0)

    S = 0
    # For discrete
    classes = Y.groupby(Y).count()
    for cat in list(classes.index):
        p = classes.loc[cat]/Y.size
        if p > 0:
            S -= p*np.log2(p)
            # print("Entropy: ", p*np.log2(p))
        else:
            pass
    return S


def information_gain(Y, attr):
    """
    Function to calculate the information gain

    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert(Y.size == attr.size)
    assert(Y.size > 0)

    # if str(attr.dtype) == "category":
    if str(attr.dtype) == "category":
        gain = entropy(Y)
        for outlook in list(attr.unique()):
            Y_sub = Y[attr == outlook]
            if Y_sub.size > 0:
                gain -= Y_sub.size/Y.size*entropy(Y_sub)
        # print("Descrete Gain: ", gain)

    else:
        sorted_attr = attr.sort_values()
        max_info = -np.inf
        value = None
        y_class = None
        low = sorted_attr.index[0]

        for high in sorted_attr.index[1:]:
            if Y[high] == y_class:
                continue
            mean_val = np.mean([sorted_attr[low], sorted_attr[high]])
            low_Y = Y[sorted_attr <= mean_val]
            high_Y = Y[sorted_attr >= mean_val]

            inf_gain = entropy(Y) - low_Y.size/Y.size * \
                entropy(low_Y) - high_Y.size/Y.size*entropy(high_Y)

            if inf_gain > max_info:
                value = mean_val
                y_class = Y[high]
                max_info = inf_gain
            low = high
        gain = (max_info, value)
    return gain


def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    cats = Y.groupby(Y).count()
    gini = 1

    for cat in list(cats.index):
        gini -= cats.loc[cat]/np.sum(cats)**2

    return gini


def gini_gain(Y, attr):
    gini_gain = 0
    if str(attr.dtype) == "category":
        gain = gini_index(Y)
        for outlook in list(attr.unique()):
            Y_sub = Y[attr == outlook]
            if Y_sub.size > 0:
                gain -= Y_sub.size/Y.size*gini_index(Y_sub)
        # print("Descrete Gain: ", gain)

    else:
        sorted_attr = attr.sort_values()
        max_info = -np.inf
        value = None
        y_class = None
        low = sorted_attr.index[0]

        for high in sorted_attr.index[1:]:
            if Y[high] == y_class:
                continue
            mean_val = np.mean([sorted_attr[low], sorted_attr[high]])
            low_Y = Y[sorted_attr < mean_val]
            high_Y = Y[sorted_attr > mean_val]

            inf_gain = gini_index(Y) - low_Y.size/Y.size * \
                gini_index(low_Y) - high_Y.size/Y.size*gini_index(high_Y)

            if inf_gain > max_info:
                value = mean_val
                y_class = Y[high]
                max_info = inf_gain
            low = high
        gini_gain = (max_info, value)

    return gini_gain


# When output is range of values
def regression_impurity(Y, attr):
    assert(Y.size == attr.size)
    assert(Y.size > 0)

    if str(attr.dtype) == "category":
        gain = np.var(Y)
        for outlook in list(attr.unique()):
            Y_sub = Y[attr == outlook]
            gain -= Y_sub.size/Y.size*np.var(Y_sub)

    else:
        sorted_attr = attr.sort_values()
        max_info = -np.inf
        value = None
        y_class = None
        low = sorted_attr.index[0]
        for high in sorted_attr.index[1:]:
            mean_val = np.mean([sorted_attr[low], sorted_attr[high]])
            low_Y = Y[sorted_attr < mean_val]
            high_Y = Y[sorted_attr > mean_val]

            inf_gain = np.var(Y) - low_Y.size/Y.size * \
                np.var(low_Y) - high_Y.size/Y.size*np.var(high_Y)

            if inf_gain > max_info:
                value = mean_val
                max_info = inf_gain
                # print(max_info)
            low = high
        gain = (max_info, value)
    return gain
