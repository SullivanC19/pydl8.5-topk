import numpy as np
import pandas as pd
from datetime import datetime
import os

from .constants import DEPTH_VALUES, MODELS

def main(dataset: str):
    from topk.globals import get_full_data, save_results

    print(f"Running Figure 6 Experiment on Dataset {dataset}...")

    results = []
    X, y = get_full_data(dataset)

    for model, runner in MODELS.items():
        print(f"Model: {model}")
        for depth in DEPTH_VALUES:
            print(f"Depth: {depth}")
            result = runner(X, y, depth)
            time = result["time"]
            timeout = result["timeout"]
            results.append({
                "model": model,
                "depth": depth,
                "time": time,
                "timeout": timeout,
            })

            print(timeout, time)

            # stop increasing depth if already out of time
            if timeout:
                break

    save_results(pd.DataFrame(results), "fig6", dataset)

