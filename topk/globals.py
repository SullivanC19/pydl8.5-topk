import os
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob
from sklearn.model_selection import train_test_split

from topk.dataloader import load_data, load_data_numerical, load_data_numerical_tt_split


CATEGORICAL_DATASETS = [
    "audiology",
    "balance-scale",
    "breast-cancer",
    "car",
    "connect-4",
    "hayes-roth",
    "hiv-1-protease",
    "kr-vs-kp",
    "led-display",
    "letter-recognition",
    "lymph",
    "molecular-biology-promoters",
    "molecular-biology-splice",
    "monks-1",
    "monks-2",
    "monks-3",
    "mushroom",
    "nursery",
    "primary-tumor",
    "rice",
    "soybean",
    "spect",
    "tic-tac-toe",
    "vote",
    "wireless-indoor",
    "yeast",
]

NUMERICAL_DATASETS = [
    "artificial-characters",
    "credit-card",
    "dry-bean",
    "electrical-grid-stability",
    "miniboone",
    "occupancy-estimation",
    "sensorless-drive-diagnosis",
    "spambase",
    "taiwanese-bankruptcy",
    "telescope",
]

TEST_TRAIN_DATASETS = [
    "avila",
    "ml-prove",
]

ALL_DATASETS = [
    "hayes-roth",
    "car",
    "connect-4",
    "hiv-1-protease",
    "kr-vs-kp",
    "letter-recognition",
    "molecular-biology-splice",
    "monks-1",
    "nursery",
    "artificial-characters",
    "credit-card",
    "dry-bean",
    "electrical-grid-stability",
    "miniboone",
    "occupancy-estimation",
    "sensorless-drive-diagnosis",
    "spambase",
    "taiwanese-bankruptcy",
    "telescope",
    "avila",
    "ml-prove",
    "tic-tac-toe",
]

EXPERIMENTS = ["fig2", "fig3", "fig4"]

CATEGORICAL_DATA_PATH = os.path.join("topk", "data", "categorical")
NUMERICAL_DATA_PATH = os.path.join("topk", "data", "numerical")
TEST_TRAIN_DATA_PATH = os.path.join("topk", "data", "numerical")
BINARY_DATA_PATH = os.path.join("topk", "data", "binary")

RESULTS_PATH = os.path.join("topk", "results", "rebuttal")
FIGURES_PATH = os.path.join("topk", "figures", "rebuttal")
TIMESTAMP_FORMAT = "%Y-%m-%d-%H:%M:%S"

SEEDS = list(range(42, 52))
TRAIN_SIZE = 0.8

def save_results(results: pd.DataFrame, experiment: str, dataset: str):
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    dir_dataset_results = os.path.join(RESULTS_PATH, experiment, dataset)
    if not os.path.exists(dir_dataset_results):
        os.makedirs(dir_dataset_results)
    df.to_csv(os.path.join(dir_dataset_results, f"results-{timestamp}.csv"))


def load_latest_results(experiment: str, dataset: str) -> pd.DataFrame:
    results_dir = os.path.join(RESULTS_PATH, experiment, dataset)
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory does not exist: {results_dir}")

    results_files = [
        os.path.basename(f) for f in
        glob(os.path.join(results_dir, 'results-*.csv'))
    ]
    timestamps = [
        datetime.strptime(f[len('results-'):-len('.csv')], TIMESTAMP_FORMAT)
        for f in results_files
    ]
    most_recent_results_file = results_files[timestamps.index(max(timestamps))]
    return pd.read_csv(os.path.join(results_dir, most_recent_results_file))


def get_data_splits(dataset: str):
    if dataset in TEST_TRAIN_DATASETS:
        data = load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, dataset, max_splits=100)
        X_train, y_train, X_test, y_test, _, _ = data
        X_train = np.array(X_train).astype(np.int32)
        X_test = np.array(X_test).astype(np.int32)
        y_train = np.array(y_train).astype(np.int32)
        y_test = np.array(y_test).astype(np.int32)
        yield X_train, X_test, y_train, y_test
        return

    # get all data
    if dataset in CATEGORICAL_DATASETS:
        data = load_data(CATEGORICAL_DATA_PATH, dataset, frac_train=1)
    elif dataset in NUMERICAL_DATASETS:
        data = load_data_numerical(NUMERICAL_DATA_PATH, dataset, frac_train=1)
    else:
        raise ValueError("Invalid dataset")

    X, y, _, _, _, _ = data
    X = np.array(X).astype(np.int32)
    y = np.array(y).astype(np.int32)

    # yield splits of data (X_train, X_test, y_train, y_test)
    for seed in SEEDS:
        yield train_test_split(X, y, train_size=TRAIN_SIZE, shuffle=True, random_state=seed)

def get_full_data(dataset: str):
    # return only training data in pre-split data
    if dataset in TEST_TRAIN_DATASETS:
        data = load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, dataset, max_splits=100)
        X_train, y_train, X_test, y_test, _, _ = data
        X_train = np.array(X_train).astype(np.int32)
        y_train = np.array(y_train).astype(np.int32)
        return X_train, y_train

    if dataset in CATEGORICAL_DATASETS:
        data = load_data(CATEGORICAL_DATA_PATH, dataset, frac_train=1)
    elif dataset in NUMERICAL_DATASETS:
        data = load_data_numerical(NUMERICAL_DATA_PATH, dataset, frac_train=1)
    else:
        raise ValueError("Invalid dataset")

    X, y, _, _, _, _ = data
    X = np.array(X).astype(np.int32)
    y = np.array(y).astype(np.int32)
    return X, y



