
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    assert(Y.size == attr.size)
    assert(Y.size > 0)

    S = 0
    # For discrete
    classes = Y.groupby(Y).count()
    for cat in list(classes.index):
        p = classes[cat]/Y.size
        S -= p*np.log2(p)
        print("Entropy: ", p*np.log2(p))

    return S


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
        gini -= cats[cat]/sum(cats)**2

    return gini


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

    if Y.dtype == "category":
        outlooks = attr.groupby(attr).count()
        gain = entropy(Y)
        for outlook in list(outlooks.index):
            Y_sub = Y[attr = outlook]
            gain - Y_sub.size/Y.size*entropy(Y_sub)
        print("Continuous Gain: ", gain)

    else:
        gain = np.var(Y)
        outlooks = attr.groupby(attr).count()
        for outlook in list(outlooks.index):
            Y_sub = Y[attr = outlook]
            gain - Y_sub.size/Y.size*np.var(Y_sub)
        print("Continuous Gain: ", gain)

    return gain
