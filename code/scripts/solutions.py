import argparse
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Q3X_DIR = "data/Q3X.csv"
Q3Y_DIR = "data/Q3y.csv"

### UTILS ###

def import_data(dir):
    with open(dir) as f:
        return np.loadtxt(f, delimiter=",")

# From neural nets lab
def plot_perceptron(ax, X, y, w):
    pos_points = X[np.where(y==1)[0]]
    neg_points = X[np.where(y==-1)[0]]
    ax.scatter(pos_points[:, 1], pos_points[:, 2], color='blue')
    ax.scatter(neg_points[:, 1], neg_points[:, 2], color='red')
    xx = np.linspace(-1,1)
    yy = -w[0]/w[2] - w[1]/w[2] * xx
    ax.plot(xx, yy, color='orange')

    ratio = (w[2]/w[1] + w[1]/w[2])
    xpt = (-1*w[0] / w[2]) * 1/ratio
    ypt = (-1*w[0] / w[1]) * 1/ratio

    ax.arrow(xpt, ypt, w[1], w[2], head_width=0.2, color='orange')
    ax.axis('equal')


### SOLUTIONS ###

def train_perceptron(X, y, max_iter=100):
    # Psuedo code following:
    # 
    # input: (x1, y1), ... ,(xn, yn)
    # initialise: w(0) = (0, 0, ... , 0) ∈ R
    # for t = 1, ... , max iter :
    #   if there is an index i such that y (w(t), x) <= 0
    #       update w
    #       t = t + 1
    #   else:
    #       output w(t), t

    np.random.seed(2)

    # Initialise w vector (1 for each iteration)
    nfeatures = X.shape[1]
    w = np.zeros((max_iter, nfeatures))
    w[0] = np.zeros(nfeatures)

    # Iterate and adjust w
    for t in range(max_iter - 1):
        
        # Dot product multipled by y
        yXw = y * (X @ w[t].T)
        mistake_idxs = np.where(yXw <= 0)[0]

        # If there are mistakes, choose a random one and 
        # update accordingly
        if mistake_idxs.size > 0:
            print(f"Mistake found at iteration {t + 1}")
            
            i = np.random.choice(mistake_idxs)
            w[t + 1] = w[t] + y[i] * X[i]

        else:
            return w[t], t + 1

    # Max iterations reached, return the latest w vector
    return w[max_iter - 1], max_iter

def q3b():
    # import data
    X = import_data(Q3X_DIR)
    y = import_data(Q3Y_DIR)

    # create perceptron
    w, nmb_iter = train_perceptron(X,y)

    # Plot
    fig, ax = plt.subplots()
    plot_perceptron(ax, X, y, w)
    ax.set_title(f"w={w}, iterations={nmb_iter}")
    plt.savefig("outputs/Q3b.png", dpi=500)

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
        "q3b": q3b
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
