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
    figure = plt.figure(layout="constrained", figsize=(8, 8))
    nrows = 2
    ncols = 2
    ax_array = figure.subplots(nrows, ncols)
    # for i, dataset in enumerate(ALL_DATASETS):
    for i, dataset in enumerate(['artificial-characters']):
        ax = ax_array[i // nrows, i % nrows]
        results = load_latest_results("fig2", dataset)

        # - circ to get hollow marker
        # - star to represent k = d
        # - legend to have Top-{k} (remove k=0)

        timeout = results.groupby(['k', 'depth'])['timeout'].agg(
            lambda arr: np.any(arr)
        ).reset_index()
        print(timeout)
        # makrers = []
        # for _, row in results.iterrows():
        #     print(row['k'])
        #     print(timeout[(timeout['k'] == row['k']) \
        #                   & (timeout['depth'] == row['depth'])]['timeout'].values[0])
        # for k in K_VALUES:
        #     for depth in DEPTH_VALUES:
        #
        # markers = [
        #     '*' if row['k'] == 0 else (
        #         '$\circ$' if (
        #             timeout[(timeout['k'] == row['k']) \
        #                     & (timeout['depth'] == row['depth'])]['timeout'].values[0]
        #         ) else 'o')
        #     for _, row in results.iterrows()
        # ]
        # print(markers)
        sns.lineplot(
            results,
            x='depth',
            y='test_acc',
            hue='k',
            estimator='mean',
            errorbar=('se', std),
            ax=ax,
            markers='*',
        )

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

        # plt.show()



    plt.show()
