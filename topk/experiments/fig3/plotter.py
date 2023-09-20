import os

import matplotlib.pyplot as plt
import seaborn as sns

from topk.globals import load_latest_results, ALL_DATASETS, FIGURES_PATH
from .constants import DEPTH_VALUES


def main():
    sns.set_style()
    figure_samples = plt.figure(layout="constrained", figsize=(12, 64))
    figure_features = plt.figure(layout="constrained", figsize=(12, 64))
    nrows = 22
    ncols = 3
    ax_array_samples = figure_samples.subplots(nrows, ncols)
    ax_array_features = figure_features.subplots(nrows, ncols)
    for i, dataset in enumerate(ALL_DATASETS):
        results = load_latest_results("fig3", dataset)
        results_samples = results[~results['timeout'] & (results['num_features'] == -1)]
        results_features = results[~results['timeout'] & (results['num_samples'] == -1)]
        for j, depth in enumerate(DEPTH_VALUES):
            ax_samples = ax_array_samples[i, j]
            ax_features = ax_array_features[i, j]

            sns.scatterplot(
                results_samples[results_samples['depth'] == depth],
                x='num_samples',
                y='time',
                hue='model',
                style='model',
                ax=ax_samples,
            )

            sns.scatterplot(
                results_features[results_features['depth'] == depth],
                x='num_features',
                y='time',
                hue='model',
                style='model',
                ax=ax_features,
            )

            ax_samples.set_title(f"{dataset} (depth = {depth})")
            ax_samples.set_xlabel('Number of Samples')
            ax_samples.set_ylabel('Time (s)')
            ax_samples.set_ylim([0, 10])

            ax_features.set_title(f"{dataset} (depth = {depth})")
            ax_features.set_xlabel('Number of Features')
            ax_features.set_ylabel('Time (s)')
            ax_features.set_ylim([0, 10])

            handles, labels = ax_features.get_legend_handles_labels()
            ax_features.get_legend().remove()
            ax_samples.get_legend().remove()

    legend_samples = figure_samples.legend(
        handles,
        labels,
        title='Model',
        loc='lower center',
        ncol=7,
        bbox_to_anchor=(0.5, -0.01),
    )
    legend_features = figure_features.legend(
        handles,
        labels,
        title='Model',
        loc='lower center',
        ncol=7,
        bbox_to_anchor=(0.5, -0.01),
    )

    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)

    fig_file_samples = os.path.join(FIGURES_PATH, "fig3-samples.pdf")
    fig_file_features = os.path.join(FIGURES_PATH, "fig3-features.pdf")
    figure_samples.savefig(fig_file_samples, format='pdf', bbox_extra_artists=(legend_samples,), bbox_inches='tight')
    figure_features.savefig(fig_file_features, format='pdf', bbox_extra_artists=(legend_features,), bbox_inches='tight')
