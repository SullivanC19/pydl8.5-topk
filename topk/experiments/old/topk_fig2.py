import os
from multiprocessing import Process
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

from tqdm import tqdm

from pydl85 import DL85Classifier, Cache_Type

from topk.dataloader import load_data, load_data_numerical, load_data_numerical_tt_split

CATEGORICAL_DATA_PATH = "topk/data/categorical"
NUMERICAL_DATA_PATH = "topk/data/numerical"
TEST_TRAIN_DATA_PATH = "topk/data/numerical"

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
    
SEEDS = list(range(42, 52))

KS = [1,2,3,4,8,12,16,0]
MAXDEPTHS = [3, 4, 5, 6, 7, 8]

TRAIN_ACC_PATH = "topk/results/accuracy/dataset={dataset}_seed={seed}-train.npy"
TEST_ACC_PATH = "topk/results/accuracy/dataset={dataset}_seed={seed}-test.npy"

TREE_AND_RESULTS_PATH = "topk/results/trees/dataset={dataset}_k={k}_maxdepth={maxdepth}_seed={seed}.pkl"

CATEGORICAL_RESULTS_PATH = "topk/results/categorical"
NUMERICAL_RESULTS_PATH = "topk/results/numerical"
TOP_K_RESULTS_FILE = "top{k}.csv"

def run_topk_experiment(dataset, data, k, maxdepth, seed):
    X_train, y_train, X_test, y_test, _, _ = data
    X_train = np.array(X_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.int32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    clf = DL85Classifier(
        k=k,
        max_depth=maxdepth,
        depth_two_special_algo=False,
        similar_lb=False,
        similar_for_branching=False,
        desc=True,
        repeat_sort=True,
        cache_type=Cache_Type.Cache_HashCover,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    pickle.dump((clf.tree_, train_acc, test_acc), open(TREE_AND_RESULTS_PATH.format(dataset=dataset, k=k, maxdepth=maxdepth, seed=seed), "wb"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("data_idx", type=int)
    parser.add_argument("seed_idx", type=int)
    parser.add_argument('-a', "--all", action="store_true")
    args = parser.parse_args()

    data_idx = args.data_idx
    seed_idx = args.seed_idx
    
    seed = SEEDS[seed_idx]

    data = None
    dataset = None
    if data_idx < len(CATEGORICAL_DATASETS):
        dataset = CATEGORICAL_DATASETS[data_idx]
        print(dataset, seed)
        data = load_data(CATEGORICAL_DATA_PATH, dataset, seed=seed)
    elif data_idx < len(CATEGORICAL_DATASETS) + len(NUMERICAL_DATASETS):
        data_idx -= len(CATEGORICAL_DATASETS)
        dataset = NUMERICAL_DATASETS[data_idx]
        print(dataset, seed)
        data = load_data_numerical(NUMERICAL_DATA_PATH, dataset, seed=seed)
    elif data_idx < len(CATEGORICAL_DATASETS) + len(NUMERICAL_DATASETS) + len(TEST_TRAIN_DATASETS):
        data_idx -= len(CATEGORICAL_DATASETS) + len(NUMERICAL_DATASETS)
        dataset = TEST_TRAIN_DATASETS[data_idx]
        print(dataset, seed)
        data = load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, dataset, max_splits=100)
        if seed != SEEDS[0]:
            exit() # train/test split won't change with seed
    else:
        raise ValueError("Invalid idx")

    trees = []
    train_accs = []
    test_accs = []

    print(f"feat: {len(data[0][0])}")
    num_feat = len(data[0][0])
    k_values = range(1, num_feat + 1) if args.all else KS

    for k in k_values:
        print(f"k={k}")
        trees.append([])
        train_accs.append([])
        test_accs.append([])
        for maxdepth in MAXDEPTHS:
            print(f"maxdepth={maxdepth}")
            # p = Process(target=run_topk_experiment, args=(dataset, data, k, maxdepth, seed))
            # p.start()
            # p.join()
            # if p.exitcode:
            #     break
            run_topk_experiment(dataset, data, k, maxdepth, seed)
