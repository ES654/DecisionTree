
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    # TODO: Write here
    return (y_hat == y).sum()/y.size


def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    pred_pos = y_hat == cls
    if sum(pred_pos) > 0:
        return (y_hat[pred_pos] == y[pred_pos]).sum()/pred_pos.sum()
    else:
        return None


def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    act_pos = y == cls
    if sum(act_pos) > 0:
        return (y_hat[act_pos] == y[act_pos]).sum()/act_pos.sum()
    else:
        return None


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return ((y-y_hat)**2).mean()**0.5


def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    return abs(y-y_hat).mean()
