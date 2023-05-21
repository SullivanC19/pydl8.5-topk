import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from topk_fig2 import TREE_AND_RESULTS_PATH, NUMERICAL_DATASETS, CATEGORICAL_DATASETS, TEST_TRAIN_DATASETS, SEEDS, KS, MAXDEPTHS

def get_results(dataset, std_dev=2, k_values=KS):
    train_acc_mean = np.zeros((len(k_values), len(MAXDEPTHS)))
    test_acc_mean = np.zeros((len(k_values), len(MAXDEPTHS)))
    train_acc_stdd = np.zeros((len(k_values), len(MAXDEPTHS)))
    test_acc_stdd = np.zeros((len(k_values), len(MAXDEPTHS)))
    for i, k in enumerate(k_values):
        print(f"k={k}")
        for j, maxdepth in enumerate(MAXDEPTHS):
            print(f"maxdepth={maxdepth}")
            train_acc = []
            test_acc = []
            for seed in SEEDS:
                path = TREE_AND_RESULTS_PATH.format(
                    dataset=dataset,
                    k=k,
                    maxdepth=maxdepth,
                    seed=seed,
                )
                if not os.path.exists(path):
                    continue
                obj = pickle.load(open(path, 'rb'))
                train_acc.append(obj[1])
                test_acc.append(obj[2])
            if (len(train_acc) == 0):
                print("No results")
                continue
            train_acc_mean[i, j] = np.mean(train_acc)
            test_acc_mean[i, j] = np.mean(test_acc)
            train_acc_stdd[i, j] = np.std(train_acc) / np.sqrt(max(1, len(train_acc) - 1)) * std_dev
            test_acc_stdd[i, j] = np.std(test_acc) / np.sqrt(max(1, len(test_acc) - 1)) * std_dev

    return train_acc_mean, test_acc_mean, train_acc_stdd, test_acc_stdd

def plot_k_vs_acc(dataset, depth_idx=0, num_features=None, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    k_values = KS if num_features is None else list(range(1, num_features + 1))
    k_values = np.array(k_values)
    k_values = k_values[k_values > 0]

    train_acc_mean, test_acc_mean, train_acc_stdd, test_acc_stdd = get_results(dataset, k_values=k_values)

    train_acc_mean, test_acc_mean, train_acc_stdd, test_acc_stdd = \
        train_acc_mean[:, depth_idx], test_acc_mean[:, depth_idx], train_acc_stdd[:, depth_idx], test_acc_stdd[:, depth_idx]
    
    nonzero = test_acc_mean > 0
    valid_k = k_values[nonzero]
    if dataset in TEST_TRAIN_DATASETS:
        ax.plot(
            valid_k, test_acc_mean[nonzero],
            label='Test',
            marker='o',
            linestyle='-',
            markeredgewidth=1,
            markersize=3,
            linewidth=1)
        ax.plot(
            valid_k, train_acc_mean[nonzero],
            label='Train',
            marker='o',
            linestyle='--',
            markeredgewidth=1,
            markersize=3,
            linewidth=1)
    else:
        print(test_acc_mean.shape)
        print(test_acc_stdd.shape)
        print(nonzero)
        print(len(valid_k))
        print()
        ax.errorbar(
            valid_k, test_acc_mean[nonzero], yerr=test_acc_stdd[nonzero],
            label='Test',
            marker='o',
            linestyle='-',
            capsize=3,
            elinewidth=1,
            markeredgewidth=1,
            markersize=3,
            linewidth=1)
        ax.errorbar(
            valid_k, train_acc_mean[nonzero], yerr=train_acc_stdd[nonzero],
            label='Train',
            marker='o',
            linestyle='--',
            capsize=3,
            elinewidth=1,
            markeredgewidth=1,
            markersize=3,
            linewidth=1)
    ax.set_xlabel("k")
    ax.set_ylabel("Test/Train Accuracy")
    ax.set_title(dataset)

def plot_depth_vs_acc(dataset, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    train_acc_mean, test_acc_mean, train_acc_stdd, test_acc_stdd = get_results(dataset)
        
    for i, k in enumerate(KS):
        nonzero = test_acc_mean[i] > 0
        acc = test_acc_mean[i, nonzero]
        std = test_acc_stdd[i, nonzero]
        valid_depths = [MAXDEPTHS[d] for d, is_valid in enumerate(nonzero) if is_valid]
        if dataset in TEST_TRAIN_DATASETS:
            ax.plot(
                valid_depths, acc,
                label=f"k={(k if k > 0 else 'd')}",
                marker='o' if k > 0 else '*',
                linestyle='-' if k > 0 else '--',
                markeredgewidth=1,
                markersize=3 if k > 0 else 8,
                linewidth=1)
        else:
            ax.errorbar(
                valid_depths, acc, yerr=std,
                label=f"k={(k if k > 0 else 'd')}",
                marker='o' if k > 0 else '*',
                linestyle='-' if k > 0 else '--',
                capsize=3,
                elinewidth=1,
                markeredgewidth=1,
                markersize=3 if k > 0 else 8,
                linewidth=1)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Test accuracy")
    ax.set_title(dataset)

if __name__ == '__main__':

    datasets = [
        ['nursery', 'occupancy-estimation', 'ml-prove', 'artificial-characters'],
        ['spambase', 'avila', 'dry-bean', 'telescope'],
        ['connect-4', 'letter-recognition', 'miniboone', 'sensorless-drive-diagnosis'],
    ]

    appendix_datasets = [
        [
            'taiwanese-bankruptcy',
            'credit-card',
            'electrical-grid-stability',
            'car',
        ],
        [
            'kr-vs-kp',
            'hiv-1-protease',
            'molecular-biology-splice',
            'monks-1',
        ],
    ]

    def build_fig2(datasets, path, figsize=(18, 9)):
        fig, ax = plt.subplots(len(datasets), len(datasets[0]), figsize=figsize)
        for i, row in enumerate(datasets):
            for j, dataset in enumerate(row):
                plot_depth_vs_acc(dataset, ax=ax[i, j])
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.tight_layout(h_pad=1.5, w_pad=0.75, rect=[0, 0.05, 1, 1])
        fig.legend(handles, labels, loc='outside lower center', ncol=8)
        plt.savefig(path, format='pdf')

    build_fig2(datasets, "topk/results/figs/fig2.pdf")
    build_fig2(appendix_datasets, "topk/results/figs/fig2-appendix.pdf", figsize=(18, 6))
    

    # for dataset in NUMERICAL_DATASETS + CATEGORICAL_DATASETS + TEST_TRAIN_DATASETS:
    #     fig, ax = plt.subplots()
    #     plot_k_vs_acc(dataset, depth_idx=0, ax=ax)
    #     plt.savefig(f"topk/results/figs/fig3/{dataset}.pdf", format='pdf', bbox_inches='tight')
    

    all_k_datasets = {
        'car': 21,
        'hayes-roth': 15,
        'tic-tac-toe': 27,
        # 'lymph': 59,
    }

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3))
    for i, dataset in enumerate(all_k_datasets):
        plot_k_vs_acc(dataset, depth_idx=0, ax=ax[i], num_features=all_k_datasets[dataset])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.5, w_pad=0.75, rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    plt.savefig("topk/results/figs/fig3-ext.pdf", format='pdf', bbox_inches='tight')
