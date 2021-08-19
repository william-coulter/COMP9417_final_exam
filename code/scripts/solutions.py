import argparse

from pandas import read_csv, concat

import math
import matplotlib.pyplot as plt
import numpy as np

Q1_DATA_DIR = "./data/Q1.csv"

### UTILS ###

def import_data(dir):
    return read_csv(dir, index_col=0)

### SOLUTIONS ###

def q1b():
    # Import data
    data = import_data(Q1_DATA_DIR)

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
        "q1b": q1b
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
