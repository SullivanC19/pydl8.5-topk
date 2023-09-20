import os

import matplotlib.pyplot as plt
import seaborn as sns

from topk.globals import load_latest_results, ALL_DATASETS, FIGURES_PATH

def main():
    sns.set_style()
    fig = plt.figure(layout="constrained", figsize=(24, 24))
    nrows = 6
    ncols = 4
    axis = fig.subplots(nrows, ncols)
    for i, dataset in enumerate(ALL_DATASETS):
        results = load_latest_results("fig6", dataset)
        ax = axis[i // ncols][i % ncols]
        sns.lineplot(
            results,
            x='depth',
            y='time',
            hue='model',
            ax=ax,
        )

        ax.set_title(f"{dataset}")
        ax.set_xlabel('Depth')
        ax.set_ylabel('Time (s)')
        ax.set_yscale('log')

        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    legend = fig.legend(
        handles,
        labels,
        title='Model',
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.5, -0.01),
    )

    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)

    fig_file = os.path.join(FIGURES_PATH, "fig6.pdf")
    fig.savefig(fig_file, format='pdf', bbox_extra_artists=(legend,), bbox_inches='tight')
