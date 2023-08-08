import os

import matplotlib.pyplot as plt
import seaborn as sns

from topk.globals import load_latest_results, ALL_DATASETS, FIGURES_PATH
from .constants import DEPTH_VALUES


def main():
    sns.set_style()
    figure = plt.figure(layout="constrained", figsize=(12, 64))
    nrows = 22
    ncols = 3
    ax_array = figure.subplots(nrows, ncols)
    for i, dataset in enumerate(['avila']): # TODO
        results = load_latest_results("fig3", dataset)
        results = results[~results['timeout'] & (results['num_features'] == -1)][['model', 'depth', 'num_samples', 'time']]
        for j, depth in enumerate(DEPTH_VALUES):
            ax = ax_array[i, j]

            depth_results = results[results['depth'] == depth]
            sns.scatterplot(
                depth_results,
                x='num_samples',
                y='time',
                hue='model',
                style='model',
                ax=ax,
            )

            ax.set_title(f"Depth = {depth}")
            ax.set_xlabel('Number of Samples')
            ax.set_ylabel('Time (s)')

    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)

    fig_file = os.path.join(FIGURES_PATH, "fig3.pdf")
    figure.savefig(fig_file, format='pdf', bbox_inches='tight')
