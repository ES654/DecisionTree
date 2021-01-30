
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tqdm import tqdm
from datetime import datetime
import os

np.random.seed(42)
num_average_time = 100

N_range = range(30, 51, 5)
M_range = range(2, 7)


def datagen(N=30, M=5, io="dido"):
    '''
    A function to generate dummy data.

    N:number of datapoints
    M: number of attributes
    io: type of data
    '''
    N = int(N)
    M = int(M)
    P = 5  # No of categories

    if io == "dido":
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif io == "diro":
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    elif io == "riro":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    else:
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")

    return X, y


def experiment(N=[30, ], M=[5, ], exp="sklearn", ios=["dido", ]):
    '''
    Creates a new experiment while varing one parameter, either M or 
    N, exp can be sklearn or mine.
    N: number of samples
    M: number of attributes
    exp: sklearn or mine
    ios: list of input outputs, can be rido, dido, riro, diro
    '''
    results = np.zeros((len(ios), len(N), len(M), 4))

    assert(len(N) == 1 or len(M) == 1)
    pbar = tqdm(total=len(ios)*len(N)*len(M))
    for io in ios:
        for n in N:
            for m in M:
                X, y = datagen(N=n, M=m, io=io)

                if exp == "sklearn":
                    if io == "diro" or io == "riro":
                        tree = DecisionTreeRegressor()
                    else:
                        tree = DecisionTreeClassifier(criterion="entropy")
                else:
                    exp = "MyTree"
                    tree = DecisionTree()
                start = datetime.now()
                tree.fit(X, y)
                end = datetime.now()
                learn = (end-start).total_seconds()

                start = datetime.now()
                tree.predict(X)
                end = datetime.now()
                predict = (end-start).total_seconds()

                results[
                    ios.index(io),
                    N.index(n),
                    M.index(m)
                ] = np.array([n, m, learn, predict])
                pbar.update(1)

    # Ploting for Learning tasks
    plt.figure()
    if len(N) > 1 or len(M) > 1:
        if len(N) > 1:
            for io in ios:
                plt.plot(results[ios.index(io), :, 0, 0],
                         results[ios.index(io), :, 0, 2],
                         label=io)
            plt.title("Learning Plot for N vs time for " + exp)
            plt.xlabel("Varying N")

        else:
            for io in ios:
                plt.plot(results[ios.index(io), 0, :, 1],
                         results[ios.index(io), 0, :, 2],
                         label=io)
            plt.title("Learning Plot for M vs time for " + exp)
            plt.xlabel("Varying M")
    plt.legend()
    plt.ylabel("Time taken in seconds")
    plt.savefig(os.path.join("exp", "learn.png"))

    # Ploting for Predicting tasks
    plt.figure()
    if len(N) > 1 or len(M) > 1:
        if len(N) > 1:
            for io in ios:
                plt.plot(results[ios.index(io), :, 0, 0],
                         results[ios.index(io), :, 0, 3],
                         label=io)
            plt.title("Prediction Plot for N vs time for " + exp)
            plt.xlabel("Varying N")

        else:
            for io in ios:
                plt.plot(results[ios.index(io), 0, :, 1],
                         results[ios.index(io), 0, :, 3],
                         label=io)
            plt.title("Prediction Plot for M vs time for " + exp)
            plt.xlabel("Varying M")
    plt.legend()
    plt.ylabel("Time taken in seconds")
    plt.savefig(os.path.join("exp", "predict.png"))
    return results


exps = ["dido", "diro", "rido", "riro"]

# Saves the plot in "exp" folder

# Can be called in the folling ways -

experiment(N=list(range(30, 81, 5)), ios=exps, exp="mine")
# experiment(N=list(range(30, 81, 5)), ios=exps, exp="sklearn")
# experiment(M=list(range(3, 14)), ios=exps, exp="mine")
# experiment(M=list(range(3, 14)), ios=exps, exp="sklearn")
