import argparse
from statistics import mean

from numpy import linspace
from pandas import read_csv, concat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

import matplotlib.pyplot as plt
import numpy as np

Q1_DATA_DIR = "./data/Q1.csv"

### UTILS ###

def import_data(dir):
    return read_csv(dir, index_col=0)

### SOLUTIONS ###

def q1a():
    # Import data
    data = import_data(Q1_DATA_DIR)

    Xtrain = data.iloc[:, :30].to_numpy()
    Ytrain = data.Y.to_numpy()

    # Set random seed
    np.random.seed(12)

    # Set B and C and p
    B = 500
    C = 1000
    p = Xtrain.shape[1]

    # Find confidence
    coef_mat = np.zeros(shape=(B,p))
    for b in range(B):
        b_sample = np.random.choice(np.arange(Xtrain.shape[0]), size=Xtrain.shape[0])
        Xtrain_b = Xtrain[b_sample]
        Ytrain_b = Ytrain[b_sample]
        mod = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xtrain_b, Ytrain_b)
        coef_mat[b,:] = mod.coef_

    means = np.mean(coef_mat, axis=0)
    lower = np.quantile(coef_mat, 0.05, axis=0)
    upper = np.quantile(coef_mat, 0.95, axis=0)

    colors = ["red" if lower[i] <= 0 and upper[i] >= 0 else "blue" for i in range(p)]

    plt.vlines(x=np.arange(1,p+1), ymin=lower, ymax=upper, colors=colors)
    plt.scatter(x=np.arange(1,p+1), y=means, color=colors)
    plt.xlabel("$Features$")
    plt.savefig("./outputs/NPBootstrap.png", dpi=400)

def q1b():
    # Import data
    data = import_data(Q1_DATA_DIR)

    Xtrain = data.iloc[:, :30].to_numpy()
    Ytrain = data.Y.to_numpy()

    # Set random seed
    np.random.seed(20)

    # Set B and C and p
    B = 500
    C = 1000
    p = Xtrain.shape[1]

    # Find confidence
    coef_mat = np.zeros(shape=(B,p))
    for b in range(B):
        b_sample = np.random.choice(np.arange(Xtrain.shape[0]), size=Xtrain.shape[0])
        Xtrain_b = Xtrain[b_sample]
        Ytrain_b = Ytrain[b_sample]
        mod = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xtrain_b, Ytrain_b)
        coef_mat[b,:] = mod.coef_

    means = np.mean(coef_mat, axis=0)
    lower = np.quantile(coef_mat, 0.05, axis=0)
    upper = np.quantile(coef_mat, 0.95, axis=0)

    colors = ["red" if lower[i] <= 0 and upper[i] >= 0 else "blue" for i in range(p)]

    plt.vlines(x=np.arange(1,p+1), ymin=lower, ymax=upper, colors=colors)
    plt.scatter(x=np.arange(1,p+1), y=means, color=colors)
    plt.xlabel("$Features$")
    plt.savefig("./outputs/NPBootstrap.png", dpi=400)

### MAIN ###

def command_line_parsing():
    parser = argparse.ArgumentParser(description="Main script for the classifier model")
    parser.add_argument(
        "--question", metavar="question", type=str, help=f"Tell the script which question to run"
    )

    return parser

if __name__ == "__main__":
    args = command_line_parsing().parse_args()

    questions = {
        "q1a": q1a,
        "q1b": q1b
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
