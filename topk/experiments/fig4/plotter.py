import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from topk.globals import load_latest_results, ALL_DATASETS, FIGURES_PATH


def main(std=2):
    sns.set_style()
    figure = plt.figure(layout="constrained", figsize=(20, 10))
    nrows = 4
    ncols = 6
    ax_array = figure.subplots(nrows, ncols)
    for i in range(len(ALL_DATASETS), nrows * ncols):
        figure.delaxes(ax_array[i // ncols, i % ncols])
    for i, dataset in enumerate(ALL_DATASETS):
        ax = ax_array[i // ncols, i % ncols]
        results = load_latest_results("fig4", dataset)

        sns.lineplot(
            results,
            x='k',
            y='train_acc',
            estimator='mean',
            errorbar=('se', std),
            ax=ax,
            color='orange',
            dashes=False,
        )

        sns.lineplot(
            results,
            x='k',
            y='test_acc',
            estimator='mean',
            errorbar=('se', std),
            ax=ax,
            color='blue',
            dashes=False,
        )

        ax.set_xlabel('K')
        ax.set_ylabel('Test/Train Accuracy')
        ax.set_title(dataset)

    train_handle = mlines.Line2D([], [], color='orange')
    test_handle = mlines.Line2D([], [], color='blue')
    legend = figure.legend(
        [train_handle, test_handle],
        ['Train', 'Test'],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, -0.08)
    )

    if not os.path.exists(FIGURES_PATH):
        os.makedirs(FIGURES_PATH)

    fig_file = os.path.join(FIGURES_PATH, "fig4.pdf")
    figure.savefig(fig_file, format='pdf', bbox_extra_artists=(legend,), bbox_inches='tight')