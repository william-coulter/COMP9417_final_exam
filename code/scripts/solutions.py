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
import itertools

### UTILS ###

def import_data(dir):
    return read_csv(dir, index_col=0)

def map_to_class(zero_or_one):
    if int(zero_or_one[0]) == 0:
        return -1
    else:
        return 1

### SOLUTIONS ###

# Code based off of NeuralLearning lab implementation
def train_perceptron(X, y, eta):
    # w = np.zeros((1, len(X[0])))           # init weight vector to 0s
    w = np.random.random((1, len(X[0])))
    nmb_iter = 0
    MAX_ITER = 10000

    for _ in range(MAX_ITER):               # termination condition (avoid running forever)
        
        nmb_iter += 1           
        
        # check which indices we make mistakes on, and pick one randomly to update
        yXw = (y * X) @ w.T
        mistake_idxs = np.where(yXw < 0)[0]
        if mistake_idxs.size > 0:
            i = np.random.choice(mistake_idxs)        # pick idx randomly
            w = w + eta * y[i] * X[i]                 # update w
            # print(f"Iteration {nmb_iter}: w = {w}")

        else: # no mistake made
            print(f"Converged after {nmb_iter} iterations")
            return

    print(f"Did not converge after {MAX_ITER} iterations")


# Generates the dataset for question 2b.
#
# Provide a list of tuples that contains all positive classes.
# Returns a completed dataset mapping all vectors in the space to either
# a positive or negative class. The dataset is returned as 2 lists being
# the set of X vectors and their corresponding Y values.
def generate_data_set(positive_classes):
    # Assume all tuples are the same length
    dimensions = len(positive_classes[0])
    domain = list(itertools.product([0, 1], repeat=dimensions))

    # Loop over domain
    Y = []
    for vector in domain:
        # If vector is in positive_classes, then mark as positive
        if vector in positive_classes:
            Y.append([1])
        else:
            Y.append([-1])

    return np.array(domain), np.array(Y)

def q2b():
    positive_classes_i = [(0,1,0), (0,1,1), (1,0,0), (1,1,1)]
    positive_classes_ii = [(0,1,1), (1,0,0), (1,1,0), (1,1,1)]
    positive_classes_iii = [(0,1,0,0), (0,1,0,1), (0,1,1,0), (1,0,0,0), (1,1,0,0), (1,1,1,0), (1,1,1,1)]
    positive_classes_iv = [(1,0,0,0,0,0,0), (1,0,0,0,0,0,1), (1,0,0,0,1,0,1)]

    all_classes = [positive_classes_i, positive_classes_ii, positive_classes_iii, positive_classes_iv]

    for c in all_classes:
        X, Y = generate_data_set(c)
        train_perceptron(X, Y, 1)


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
        "q2b": q2b
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
