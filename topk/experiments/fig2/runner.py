import numpy as np
import pandas as pd
from datetime import datetime
import os

from topk.algorithms.run_topk import search

from .constants import K_VALUES, DEPTH_VALUES, TIME_LIMIT

def main(dataset: str):
    from topk.globals import get_data_splits, save_results

    print(f"Running Figure 2 Experiment on Dataset {dataset}...")

    results = []
    split_idx = 0
    for data in get_data_splits(dataset):
        print(f"Split: {split_idx}")
        for k in K_VALUES:
            print(f"K: {k}")
            for depth in DEPTH_VALUES:
                print(f"Depth: {depth}")
                X_train, X_test, y_train, y_test = data
                result = search(
                    X_train,
                    y_train,
                    k=k,
                    max_depth=depth,
                    time_limit=TIME_LIMIT,
                )

                tree = result["tree"]
                time = result["time"]
                timeout = result["timeout"]
                train_acc = np.count_nonzero(tree.predict(X_train) == y_train) / y_train.shape[0]
                test_acc = np.count_nonzero(tree.predict(X_test) == y_test) / y_test.shape[0]

                results.append({
                    "split_idx": split_idx,
                    "k": k,
                    "depth": depth,
                    "tree": str(tree),
                    "time": time,
                    "timeout": timeout,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                })

                print(timeout, test_acc)
        split_idx += 1

    save_results(pd.DataFrame(results), "fig2", dataset)

