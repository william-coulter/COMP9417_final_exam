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
def plot_perceptron(ax, X, y, w, v=None):
    pos_points = X[np.where(y==1)[0]]
    neg_points = X[np.where(y==-1)[0]]
    ax.scatter(pos_points[:, 1], pos_points[:, 2], color='blue')
    ax.scatter(neg_points[:, 1], neg_points[:, 2], color='red')

    xx = np.linspace(-1.5,0)
    yy = -w[0]/w[2] - w[1]/w[2] * xx
    ax.plot(xx, yy, color='orange')

    ratio = (w[2]/w[1] + w[1]/w[2])
    xpt = (-1*w[0] / w[2]) * 1/ratio
    ypt = (-1*w[0] / w[1]) * 1/ratio

    ax.arrow(xpt, ypt, w[1], w[2], head_width=0.2, color='orange')
    ax.axis('equal')

    # For some of the winnow algorithms
    if v is not None:

        xx = np.linspace(-1.5,0)
        yy = -v[0]/v[2] - v[1]/v[2] * xx
        # ax.plot(xx, yy, color='purple')

        ratio = (v[2]/v[1] + v[1]/v[2])
        xpt = (-1*v[0] / v[2]) * 1/ratio
        ypt = (-1*v[0] / v[1]) * 1/ratio

        ax.arrow(xpt, ypt, v[1], v[2], head_width=0.2, color='purple')


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
    # w[0] = np.zeros(nfeatures)

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

def train_positive_winnow(X, y, max_iter=100):

    # input: (x1, y1), ... ,(xn, yn)
    # initialise: w[0] = (1, 1, ... , 1) ∈ R
    #
    # for t = 1, ... , max iter:
    #
    #   (Check if there are any mistakes according to the current decision function)
    #   if there is an index i s.t: y[i] * [dotproduct(w(t), x[i]) - threshold] <= 0:
    #
    #       (Check if promotion or demotion)
    #       multiplier = alpha
    #       if y[i] < 0:
    #           multipler = beta
    #
    #       (Update the weight vector)
    #       for j in 0 .. length(w[t]):
    #
    #           (Check if the corresponding feature is strictly > 0)
    #           if x[i][j] > 0:
    #               w[t+1][j] = w[t][j] * multiplier
    #           else:
    #               w[t+1][j] = w[t][j]
    #
    #   (No errors and the algorithm has converged)
    #   else:
    #       output w[t], t

    np.random.seed(2)

    promotion = 1.5
    demotion = 0.5
    threshold = 1.0
    initialisation = 1.0

    # Initialise w vector for each iteration
    nfeatures = X.shape[1]
    w = np.full(shape=(max_iter, nfeatures), fill_value=initialisation)

    # Iterate and adjust w
    for t in range(max_iter - 1):

        # Calculate any mistakes according to the decision function at w[t]
        decision_matrix = X @ w[t].T - threshold
        mistakes_matrix = y * decision_matrix
        mistake_idxs = np.where(mistakes_matrix <= 0)[0]

        # If there are mistakes, choose a random one and
        # update accordingly
        if mistake_idxs.size > 0:
            print(f"Mistake found at iteration {t + 1}")
            i = np.random.choice(mistake_idxs)

            multiplier = promotion
            if y[i] < 0:
                multiplier = demotion

            for j in range(len(w[t])):
                # Check if the corresponding feature is strictly > 0)
                if X[i][j] > 0:
                    w[t+1][j] = w[t][j] * multiplier
                else:
                    w[t+1][j] = w[t][j]

            print(f"old w: {w[t]}")
            print(f"new w: {w[t+1]}")


        else:
            # Converged
            return w[t], t + 1

    # Max iterations reached, return the latest w vector
    return w[max_iter - 1], max_iter

def train_balanced_winnow(X, y, max_iter=100):

    # input: (x1, y1), ... ,(xn, yn)
    # initialise: v[0] = (0+, 0+,..., 0+) ∈ R
    # initialise: u[0] = (0-, 0-,..., 0-) ∈ R
    #
    # for t = 1, ... , max iter:
    #
    #   (Check if there are any mistakes according to the current decision function)
    #   if there is an index i s.t:
    #               y[i] * [dotproduct(u[t], x[i]) - dotproduct(v[t], x[i]) - threshold] <= 0:
    #
    #       (Check if promotion or demotion)
    #       u_multiplier = alpha
    #       v_multiplier = beta
    #       if y[i] < 0:
    #           u_multiplier = beta
    #           v_multiplier = alpha
    #
    #       (Update the positive and negative models)
    #       u[t+1] = u[t] * u_multiplier
    #       v[t+1] = v[t] * v_multiplier
    #
    #   (No errors and the algorithm has converged)
    #   else:
    #       output w[t], t

    np.random.seed(2)

    promotion = 1.5
    demotion = 0.5
    threshold = 1.0

    # Initialise vectors for each iteration
    nfeatures = X.shape[1]
    u = np.full(shape=(max_iter, nfeatures), fill_value=2.0)
    v = np.full(shape=(max_iter, nfeatures), fill_value=1.0)

    # Iterate and adjust w
    for t in range(max_iter - 1):

        # Calculate any mistakes according to the decision function at w[t]
        decision_matrix = X @ u[t].T - X @ v[t].T -threshold
        mistakes_matrix = y * decision_matrix
        mistake_idxs = np.where(mistakes_matrix <= 0)[0]

        # If there are mistakes, choose a random one and
        # update accordingly
        if mistake_idxs.size > 0:
            print(f"Mistake found at iteration {t + 1}")
            i = np.random.choice(mistake_idxs)

            u_multiplier = promotion
            v_multiplier = demotion
            if y[i] < 0:
                u_multiplier = demotion
                v_multiplier = promotion

            # No need to check if the corresponding feature is strictly > 0
            u[t+1] = u[t] * u_multiplier
            v[t+1] = v[t] * v_multiplier

            print(f"v: {v[t]}")
            print(f"u: {u[t]}")

        else:
            # Converged
            return u[t], v[t], t + 1

    # Max iterations reached, return the latest w vector
    return u[max_iter - 1], v[max_iter - 1], max_iter

def train_modified_winnow(X, y, max_iter=100):

    # input: (x1, y1), ... ,(xn, yn)
    # initialise: v[0] = (0+, 0+,..., 0+) ∈ R
    # initialise: u[0] = (0-, 0-,..., 0-) ∈ R
    #
    # for t = 1, ... , max iter:
    #
    #   (Check if there are any mistakes according to the current decision function)
    #   if there is an index i s.t:
    #               y[i] * [dotproduct(u[t], x[i]) - dotproduct(v[t], x[i]) - threshold] <= M:
    #
    #       for j = 0, ... , number of features
    #            (Check if promotion or demotion)
    #            u_multiplier = alpha * (1 + x[i][j])
    #            v_multiplier = beta * (1 - x[i][j])
    #
    #            if y[i] < 0:
    #                u_multiplier = beta * (1 - x[i][j])
    #                v_multiplier = alpha * (1 + x[i][j])
    #
    #            (Update the positive and negative models)
    #            u[t+1][j] = u[t][j] * u_multiplier
    #            v[t+1][j] = v[t][j] * v_multiplier
    #
    #   (No errors and the algorithm has converged)
    #   else:
    #       output w[t], t

    np.random.seed(2)

    promotion = 1.5
    demotion = 0.5
    threshold = 1.0
    margin = 1.0

    # Initialise vectors for each iteration
    nfeatures = X.shape[1]
    u = np.full(shape=(max_iter, nfeatures), fill_value=2.0)
    v = np.full(shape=(max_iter, nfeatures), fill_value=1.0)

    # Iterate and adjust w
    for t in range(max_iter - 1):

        # Calculate any mistakes according to the decision function at w[t]
        decision_matrix = X @ u[t].T - X @ v[t].T -threshold
        mistakes_matrix = y * decision_matrix
        mistake_idxs = np.where(mistakes_matrix <= margin)[0]

        # If there are mistakes, choose a random one and
        # update accordingly
        if mistake_idxs.size > 0:
            print(f"Mistake found at iteration {t + 1}")
            i = np.random.choice(mistake_idxs)

            for j in range(len(X[i])):
                u_multiplier = promotion * (1 + X[i][j])
                v_multiplier = demotion + (1 - X[i][j])

                if y[i] < 0:
                    u_multiplier = demotion + (1 - X[i][j])
                    v_multiplier = promotion * (1 + X[i][j])

                # No need to check if the corresponding feature is strictly > 0
                u[t+1][j] = u[t][j] * u_multiplier
                v[t+1][j] = v[t][j] * v_multiplier

            print(f"v: {v[t]}")
            print(f"u: {u[t]}")

        else:
            # Converged
            return u[t], v[t], t + 1

    # Max iterations reached, return the latest w vector
    return u[max_iter - 1], v[max_iter - 1], max_iter

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

def q3c():
    # import data
    X = import_data(Q3X_DIR)
    y = import_data(Q3Y_DIR)

    # create perceptron
    w, nmb_iter = train_positive_winnow(X,y)

    # Plot
    fig, ax = plt.subplots()
    plot_perceptron(ax, X, y, w)
    ax.set_title(f"w={w}, iterations={nmb_iter}")
    plt.savefig("outputs/Q3c.png", dpi=500)

def q3d():
    # import data
    X = import_data(Q3X_DIR)
    y = import_data(Q3Y_DIR)

    # create perceptron
    u, v, nmb_iter = train_balanced_winnow(X,y)

    # Plot
    fig, ax = plt.subplots()
    plot_perceptron(ax, X, y, u+v)
    ax.set_title(f"u+v={u+v}\niterations={nmb_iter}")
    plt.xlim([-0.5,1.5])
    plt.ylim([-1,2])
    plt.savefig("outputs/Q3d.png", dpi=500)

def q3e():
    # import data
    X = import_data(Q3X_DIR)
    y = import_data(Q3Y_DIR)

    # create perceptron
    u, v, nmb_iter = train_modified_winnow(X,y)

    # Plot
    fig, ax = plt.subplots()
    plot_perceptron(ax, X, y, u+v)
    ax.set_title(f"u+v={u+v}\niterations={nmb_iter}")
    plt.xlim([-0.5,1.5])
    plt.ylim([-1,2])
    plt.savefig("outputs/Q3e.png", dpi=500)

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
        "q3b": q3b,
        "q3c": q3c,
        "q3d": q3d,
        "q3e": q3e
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
