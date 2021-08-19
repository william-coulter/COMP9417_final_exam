import argparse

from pandas import read_csv, concat
from sklearn.linear_model import LinearRegression

import math
import matplotlib.pyplot as plt
import numpy as np

Q4_TRAIN_DATA_DIR = "./data/Q4_train.csv"
Q4_TEST_DATA_DIR = "./data/Q4_test.csv"

### UTILS ###

def import_data(dir):
    return read_csv(dir, index_col=0)

### SOLUTIONS ###

def q4b():
    # Psuedo code for algorithm that minimises the loss function
    # I really don't know...
    pass

def total_loss(X, y, Z, models):
    loss = 0
    M = len(models)
    n = X.shape[0]

    for j in range(M):
        model = models[j]

        for i in range(n):
            # If the observation is not partitioned to model "j", we skip
            if Z[i] != j:
                continue

            # Get penalty of the coefficients for this model on this observation
            observation = X[i]
            actual = y[i]
            prediction = model.predict(np.array([observation]))
            coef_penalty = (prediction[0] - actual)**2

            # Cannot know partition_penalty since partition predictions
            # are not provided to this function. Not really sure of the
            # point of this function. Just set penalty to 0.
            partition_penalty = 0

            loss += coef_penalty + partition_penalty

    return loss

def q4c():
    data = import_data(Q4_TRAIN_DATA_DIR)
    Xtrain = np.array(data.drop(columns=["Y"]))
    ytrain = np.array(data["Y"])

    mod = LinearRegression().fit(Xtrain, ytrain)
    Z = np.zeros(shape=Xtrain.shape[0]) # all points would belong to a single partition.
    print(total_loss(Xtrain, ytrain, Z, [mod])) # outputs 298.328178158043

def find_partitions(X, y, models):
    M = len(models)
    n = X.shape[0]

    Z = []
    for i in range(n):
        observation = X[i]

        # predict this observation on each model. Whichever model
        # predicted the closest to the actual we assume this
        # datapoint belongs to this model.
        pred_diffs = []
        for model in models:
            pred = model.predict(np.array([observation]))
            actual = y[i]

            # Could also square this to get a positive number
            pred_diffs.append(abs(pred - actual))

        # Append index of smallest value to list
        Z.append(pred_diffs.index(min(pred_diffs)))

    return np.array(Z)

def q4d():
    data = import_data(Q4_TEST_DATA_DIR)
    Xtest = np.array(data.drop(columns=["Y"]))
    ytest = np.array(data["Y"])

    # Run with one model
    mod = LinearRegression().fit(Xtest, ytest)

    Z = find_partitions(Xtest, ytest, [mod])
    print(total_loss(Xtest, ytest, Z, [mod]))

# So that we don't have to train thousands of models
def generate_Z(M, n):

    # np.tile is such a misdirect, you have to find how many
    # times you want to repeat and then append two lists together.
    #
    # Sorry, I've been going at this exam for years and am getting
    # to the end of a tether.
    Z = []
    for i in range(n):
        Z.append(i % M)

    return np.array(Z)

def q4e():
    # import train and test data
    data_train = import_data(Q4_TRAIN_DATA_DIR)
    Xtrain = np.array(data_train.drop(columns=["Y"]))
    ytrain = np.array(data_train["Y"])

    data_test = import_data(Q4_TEST_DATA_DIR)
    Xtest = np.array(data_test.drop(columns=["Y"]))
    ytest = np.array(data_test["Y"])

    # model
    model = LinearRegression()

    # Iterate over values of M
    train_losses = []
    test_losses = []

    print("M  |        train        |        test          |")
    MAX_M = 30
    for M in range(1,MAX_M + 1):
        # Partition data (done in a mechanical way for exam testing)
        Ztrain = generate_Z(M, Xtrain.shape[0])
        Ztest = generate_Z(M, Xtest.shape[0])

        # Train models on their own data
        train_models = []
        for i in range(M):
            indices = np.where(Ztrain == i)
            fitted = model.fit(Xtrain[indices], ytrain[indices])
            train_models.append(fitted)

        test_models = []
        for i in range(M):
            indices = np.where(Ztest == i)
            fitted = model.fit(Xtest[indices], ytest[indices])
            test_models.append(fitted)

        # Calculate loss
        train_loss = total_loss(Xtrain, ytrain, Ztrain, train_models)
        test_loss = total_loss(Xtest, ytest, Ztest, test_models)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if M % 5 == 0:
            padding=""
            if M == 5:
                padding = " "
            print(F"{M}{padding} |  {train_loss}  |  {test_loss}  |")

    # Plot everything
    X = list(range(1, MAX_M + 1))
    plt.plot(X, train_losses, color="skyblue", label="train losses", linewidth=2)
    plt.plot(X, test_losses, color="olive", label="test losses", linewidth=2)
    plt.legend()
    plt.savefig("outputs/q4e.png")


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
        "q4b": q4b,
        "q4c": q4c,
        "q4d": q4d,
        "q4e": q4e
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
