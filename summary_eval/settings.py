import random
import numpy as np
import os


random.seed(42)
np.random.seed(42)

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))  # e.g. C:\Users\user\summary-eval
DATA_DIR = "data"
PROMPTS_TRAIN_PATH = os.path.join(ROOT_PATH, DATA_DIR, "prompts_train.csv")
SUMMARIES_TRAIN_PATH = os.path.join(ROOT_PATH, DATA_DIR, "summaries_train.csv")

# Cross validation splits
# Use 10x10 cross validation
N_RUNS = 10
N_FOLDS = 10

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2
# Testing to be done on Kaggle
