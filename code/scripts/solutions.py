import argparse
from statistics import mean

from numpy import linspace
from pandas import read_csv, concat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

import math
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
    lower = np.quantile(coef_mat, 0.10, axis=0)
    upper = np.quantile(coef_mat, 0.90, axis=0)

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

    # Train initial model
    mod = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xtrain, Ytrain)
    B0 = mod.intercept_[0]
    coefs_sum = sum(mod.coef_[0])
    # print(B0)
    # print(coefs_sum)
    # print(math.exp(B0 + coefs_sum))
    # print(mod.coef_[0])

    # Calculate p value for bernoulli distribution
    # TODO: Probability here is too high
    prob = math.exp(B0 + coefs_sum) / (1 + math.exp(B0 + coefs_sum))
    # print(prob)

    # Find confidence
    coef_mat = np.zeros(shape=(B,p))
    for b in range(B):
        # Get X values randomly as before. Here, n = Xtrain.shape[0].
        b_sample = np.random.choice(np.arange(Xtrain.shape[0]), size=Xtrain.shape[0])
        Xtrain_b = Xtrain[b_sample]

        # The Ytrain_b must come from a bernoulli distribution
        Ytrain_b = np.random.binomial(n=1, p=prob, size=Xtrain.shape[0])
        print(Ytrain_b)

        # Train new model
        mod_b = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xtrain_b, Ytrain_b)

        # store coefficients
        coef_mat[b,:] = mod_b.coef_

    means = np.mean(coef_mat, axis=0)
    lower = np.quantile(coef_mat, 0.10, axis=0)
    upper = np.quantile(coef_mat, 0.90, axis=0)

    colors = ["red" if lower[i] <= 0 and upper[i] >= 0 else "blue" for i in range(p)]

    plt.vlines(x=np.arange(1,p+1), ymin=lower, ymax=upper, colors=colors)
    plt.scatter(x=np.arange(1,p+1), y=means, color=colors)
    plt.xlabel("$Features$")
    plt.savefig("./outputs/NPParameterisedBootstrap.png", dpi=400)

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
