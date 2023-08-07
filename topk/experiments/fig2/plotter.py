import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from topk.globals import load_latest_results, ALL_DATASETS
from .constants import K_VALUES, DEPTH_VALUES


def main(std=2):
    sns.set_style()
    figure = plt.figure(layout="constrained", figsize=(16, 12))
    nrows = 6
    ncols = 4
    ax_array = figure.subplots(nrows, ncols)
    # for i, dataset in enumerate(ALL_DATASETS):
    for i, dataset in enumerate(['artificial-characters']):
        ax = ax_array[i // nrows, i % nrows]
        results = load_latest_results("fig2", dataset)

        sns.lineplot(results, x='depth', y='test_acc', estimator='mean', ax=ax)

        # data = results.groupby(['k', 'depth'])['test_acc'].agg(
        #     acc=lambda arr: np.mean(arr), 'err':
        #         lambda arr: np.sqrt(arr) / np.sqrt(len(arr)) * std}),
        # ).reset_index()
        #
        # for k in K_VALUES:
        #     acc = np.array(results[results['k'] == k].groupby('depth')['test_acc'].agg('mean'))
        #     print(acc)
        #     exit()
        #     err = np.array(grouped.agg('std')) / np.sqrt(len(acc)) * std
        #     marker = "*" if k == 0 else 'o'
        #     k = 'd' if k > 0 else str(k)
        #     plt.errorbar(
        #         DEPTH_VALUES[~data['timeout']],
        #         data['acc'][~data['timeout']],
        #         yerr=data['err'],
        #         marker=marker,
        #         fillstyle='full',
        #         keyword=f"Top-{k}",
        #     )
        #     plt.errorbar(
        #         DEPTH_VALUES[data['timeout']],
        #         data['acc'][data['timeout']],
        #         yerr=data['err'],
        #         marker=marker,
        #         fillstyle='none',
        #         keyword=f"Top-{k}",
        #     )

        plt.show()



    plt.show()
