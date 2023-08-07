import os
from multiprocessing import Process

from prettytable import PrettyTable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from pydl85 import DL85Classifier, Cache_Type, Wipe_Type

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
    # "artificial-characters",
    # "credit-card",
    # "dry-bean",
    # "electrical-grid-stability",
    # "miniboone",
    # "occupancy-estimation",
    # "sensorless-drive-diagnosis",
    # "spambase",
    # "taiwanese-bankruptcy",
    # "telescope",
]

TEST_TRAIN_DATASETS = [
    # "avila",
    # "ml-prove",
]

CATEGORICAL_RESULTS_PATH = "topk/results/categorical"
NUMERICAL_RESULTS_PATH = "topk/results/numerical"
TOP_K_RESULTS_FILE = "top{k}.csv"
k_values = [2,3,4]
max_depths = [3,4,5,6,7,8]

TRAIN_ACC_PATH = "topk/results/accuracy/{dataset}-train.npy"
TEST_ACC_PATH = "topk/results/accuracy/{dataset}-test.npy"

def get_categorical_data(dataset):
    return load_data(CATEGORICAL_DATA_PATH, dataset)

def get_numerical_data(dataset):
    return load_data_numerical(NUMERICAL_DATA_PATH, dataset)

def get_test_train_data(dataset):
    return load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, dataset, max_splits=100)

def top_k_accuracy(data):
    X_train, y_train, X_test, y_test, _, _ = data
    
    X_train = np.array(X_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.int32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    train_acc = np.zeros((len(k_values), len(max_depths)))
    test_acc = np.zeros((len(k_values), len(max_depths)))

    for i, k in tqdm(list(enumerate(k_values))[::-1]):
        for j, d in list(enumerate(max_depths))[::-1]:
            clf = DL85Classifier(
                k=k,
                max_depth=d,
                depth_two_special_algo=False,
                similar_lb=False,
                similar_for_branching=False,
                desc=True,
                repeat_sort=True,
                cache_type=Cache_Type.Cache_HashCover,
            )
            clf.fit(X_train, y_train)

            if clf.timeout_:
                return None, None
            else:
                test_acc[i, j] = clf.score(X_test, y_test)
                train_acc[i, j] = clf.score(X_train, y_train)

    return train_acc, test_acc

def func(dataset, f_get_data):
    data = f_get_data(dataset)
    train_acc, test_acc = top_k_accuracy(data)
    np.save(TRAIN_ACC_PATH.format(dataset=dataset), train_acc)
    np.save(TEST_ACC_PATH.format(dataset=dataset), test_acc)

if __name__ == '__main__':

    processes = \
        [Process(target=func, args=(dataset, get_categorical_data)) for dataset in CATEGORICAL_DATASETS] \
        + [Process(target=func, args=(dataset, get_numerical_data)) for dataset in NUMERICAL_DATASETS] \
        + [Process(target=func, args=(dataset, get_test_train_data)) for dataset in TEST_TRAIN_DATASETS] \
        
    # bandaid fix to avoid segfaut errors interrupting experiment
    for p in processes:
        p.start()
        p.join()

    categorical_results = []
    for k in k_values:
        path = os.path.join(CATEGORICAL_RESULTS_PATH, TOP_K_RESULTS_FILE.format(k=k))
        categorical_results.append(pd.read_csv(path))
    
    numerical_results = []
    for k in k_values:
        path = os.path.join(NUMERICAL_RESULTS_PATH, TOP_K_RESULTS_FILE.format(k=k))
        numerical_results.append(pd.read_csv(path))

    unable_to_compute = []
    results_missing = []

    total_comparisons = 0

    thresholds = [0, 0.0001, 0.001, 0.01, 0.1]
    total_train_within = np.zeros(len(thresholds))
    total_test_within = np.zeros(len(thresholds))
    max_train_diff = 0.0
    max_test_diff = 0.0

    total_abs_train_diff = 0.0
    total_abs_test_diff = 0.0

    total_train_diff = 0.0
    total_test_diff = 0.0

    for dataset in CATEGORICAL_DATASETS + NUMERICAL_DATASETS + TEST_TRAIN_DATASETS:
        train_path = TRAIN_ACC_PATH.format(dataset=dataset)
        test_path = TEST_ACC_PATH.format(dataset=dataset)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            unable_to_compute.append(dataset)
            continue

        train_acc = np.load(train_path, allow_pickle=True)
        test_acc = np.load(test_path, allow_pickle=True)

        if (train_acc == None).any() or (test_acc == None).any():
            unable_to_compute.append(dataset)
            continue

        base_train_acc = np.zeros((len(k_values), len(max_depths)))
        base_test_acc = np.zeros((len(k_values), len(max_depths)))

        results = categorical_results if dataset in CATEGORICAL_DATASETS else numerical_results

        for i, k in enumerate(k_values):
            df = results[i]
            df = df[df["Dataset"] == dataset]
            df = df[df["Impurity"] == "entropy"]
            if df.shape[0] == 0:
                results_missing.append(dataset)
                break
            base_train_acc[i] = df["Train"].to_numpy()
            base_test_acc[i] = df["Test"].to_numpy()

        if dataset in results_missing:
            continue

        print()
        print(dataset)

        total_train_diff += np.sum(train_acc - base_train_acc)
        total_test_diff += np.sum(test_acc - base_test_acc)

        train_table = PrettyTable()
        train_table.field_names = ["k"] + [f"d={d}" for d in max_depths]
        for i, k in enumerate(k_values):
            train_table.add_row([k] + list(np.round(train_acc[i] - base_train_acc[i], 3)))

        test_table = PrettyTable()
        test_table.field_names = ["k"] + [f"d={d}" for d in max_depths]
        for i, k in enumerate(k_values):
            test_table.add_row([k] + list(np.round(test_acc[i] - base_test_acc[i], 3)))

        print("Train Accuracy Diff")
        print(train_table)

        print("Test Accuracy Diff")
        print(test_table)

        total_abs_train_diff = np.sum(np.abs(train_acc - base_train_acc))
        total_abs_test_diff = np.sum(np.abs(test_acc - base_test_acc))

        for l, thr in enumerate(thresholds):
            total_train_within[l] += np.sum(np.abs(train_acc - base_train_acc) <= thr)
            total_test_within[l] += np.sum(np.abs(test_acc - base_test_acc) <= thr)

        total_comparisons += len(k_values) * len(max_depths)

    print()

    print("Total Comparisons: ", total_comparisons)
    print(f"Average Train Diff: {total_train_diff / total_comparisons}")
    print(f"Average Test Diff: {total_test_diff / total_comparisons}")

    print("Thresholds: ", thresholds)

    print("Percentage of Train Values in Thresholds: ", total_train_within / total_comparisons)
    print("Percentage of Test Values in Thresholds: ", total_test_within / total_comparisons)

    print("Average Train Value Diff: ", total_abs_train_diff / total_comparisons)
    print("Average Test Value Diff: ", total_abs_test_diff / total_comparisons)

    print("Unable to compute: ", unable_to_compute)
    print("Results missing: ", results_missing)
    