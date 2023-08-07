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
        results = load_latest_results("fig2", dataset)
        results['k'] = results['k'].replace(0, 32)

        sns.lineplot(
            results[results['k'] != 32],
            x='depth',
            y='test_acc',
            hue='k',
            hue_norm=(0, 32),
            estimator='mean',
            errorbar=('se', std),
            ax=ax,
            marker='o',
            dashes=False,
        )
        sns.lineplot(
            results[results['k'] == 32],
            x='depth',
            y='test_acc',
            hue='k',
            hue_norm=(0, 32),
            estimator='mean',
            errorbar=('se', std),
            ax=ax,
            marker='d',
            dashes=False,
        )

        # sns.lineplot(
        #     results[(results['k'] != 32) & (results['timeout'])],
        #     x='depth',
        #     y='test_acc',
        #     style='k',
        #     estimator='mean',
        #     errorbar=None,
        #     ax=ax,
        #     marker='o',
        #     color='orange',
        #     dashes=False,
        #     linestyle='',
        # )

        sns.lineplot(
            results[(results['k'] == 32) & (results['timeout'])],
            x='depth',
            y='test_acc',
            style='k',
            estimator='mean',
            errorbar=None,
            ax=ax,
            marker='d',
            color='orange',
            dashes=False,
            linestyle='',
        )

        ax.set_title(dataset)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Test Accuracy')

        # Manually rewrite k = 0 to k = d
        handles, labels = ax.get_legend_handles_labels()
        labels[5] = 'd'
        # ax.legend(handles[:6], labels[:6], title='k')

        # Disable the subfigure legend, we'll add a global legend later
        ax.get_legend().remove()

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

        # Disable the subfigure legend, we'll add a global legend later
        ax.get_legend().remove()

    # The [:6] prevents duplicate labels from showing up more than once
    # handles = handles[:6]
    # labels = labels[:6]
    # for i in range(len(handles)):
    #     handles[i] = mlines.Line2D([], [], color=handles[i].get_color(), marker='d' if labels[i] == 'd' else 'o')
    # lgd = figure.legend(
    #     handles,
    #     labels,
    #     title='k',
    #     loc='lower center',
    #     ncol=6,
    #     bbox_to_anchor=(0.5, -0.08)
    # )
    #
    # if not os.path.exists(FIGURES_PATH):
    #     os.makedirs(FIGURES_PATH)
    lgd = figure.legend(handles[:6], labels[:6], title='k', loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.08))
    figure.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')

    fig_file = os.path.join(FIGURES_PATH, "fig2.pdf")
    figure.savefig(fig_file, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
