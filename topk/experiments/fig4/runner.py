import numpy as np
import pandas as pd
from datetime import datetime
import os

from topk.algorithms.run_topk import search

from .constants import DEPTH, TIME_LIMIT


def main(dataset: str):
    from topk.globals import get_data_splits, RESULTS_PATH, TIMESTAMP_FORMAT

    print(f"Running Figure 4 Experiment on Dataset {dataset}...")

    results = []
    split_idx = 0
    for data in get_data_splits(dataset):
        print(f"Split: {split_idx}")
        num_features = data[0].shape[1]
        for k in range(1, num_features + 1):
            print(f"K: {k}")
            X_train, X_test, y_train, y_test = data
            result = search(
                X_train,
                y_train,
                k=k,
                max_depth=DEPTH,
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
                "depth": DEPTH,
                "tree": str(tree),
                "time": time,
                "timeout": timeout,
                "train_acc": train_acc,
                "test_acc": test_acc,
            })
        split_idx += 1

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    dir_dataset_results = os.path.join(RESULTS_PATH, "fig4", dataset)
    if not os.path.exists(dir_dataset_results):
        os.makedirs(dir_dataset_results)
    df.to_csv(os.path.join(dir_dataset_results, f"results-{timestamp}.csv"))

