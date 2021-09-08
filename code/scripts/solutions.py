import argparse
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Q3X_DIR = "data/Q3X.csv"
Q3Y_DIR = "data/Q3y.csv"

### UTILS ###

def import_data(dir, columns):
    return pd.read_csv(dir, columns=columns)

### SOLUTIONS ###



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
        "q2b": q2b,
        "q2d": q2d
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
