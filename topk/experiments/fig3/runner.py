import numpy as np
import pandas as pd
from datetime import datetime
import os

from .constants import DEPTH_VALUES, RUNS_PER_DATASET, MODELS

def main(dataset: str):
    from topk.globals import get_full_data, save_results

    print(f"Running Figure 3 Experiment on Dataset {dataset}...")

    results = []
    X, y = get_full_data(dataset)
    total_samples = X.shape[0]
    total_features = X.shape[1]

    num_samples_values = np.unique(np.round(np.linspace(10, total_samples, num=RUNS_PER_DATASET)).astype(np.int32))
    num_features_values = np.unique(np.round(np.linspace(10, total_features, num=RUNS_PER_DATASET)).astype(np.int32))

    for model, runner in MODELS.items():
        print(f"Model: {model}")
        for depth in DEPTH_VALUES:
            print(f"Depth: {depth}")
            for num_samples in num_samples_values:
                print(f"Num Samples: {num_samples}")
                X_sub = X[:num_samples, :]
                y_sub = y[:num_samples]

                result = runner(X_sub, y_sub, depth)
                time = result["time"]
                timeout = result["timeout"]
                results.append({
                    "model": model,
                    "depth": depth,
                    "time": time,
                    "num_samples": num_samples,
                    "num_features": -1,  # all features
                    "timeout": timeout,
                })

                print(timeout, time)

                # stop increasing number of samples if already out of time
                if timeout:
                    break

            for num_features in num_features_values:
                print(f"Num Features: {num_features}")
                X_sub = X[:, :num_features]
                y_sub = y

                result = runner(X_sub, y_sub, depth)
                time = result["time"]
                timeout = result["timeout"]
                results.append({
                    "model": model,
                    "depth": depth,
                    "time": time,
                    "num_samples": -1,  # all samples
                    "num_features": num_features,
                    "timeout": timeout,
                })

                print(timeout, time)

                # stop increasing number of features if already out of time
                if timeout:
                    break

    save_results(pd.DataFrame(results), "fig3", dataset)

