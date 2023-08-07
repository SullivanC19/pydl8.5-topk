import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from run_topk import CATEGORICAL_DATASETS, NUMERICAL_DATASETS, TEST_TRAIN_DATASETS

DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TOPKS = [1, 2, 4, 8, 12, 16]
# DATASETS = ['connect-4', 'artificial-characters']
DATASETS = CATEGORICAL_DATASETS + NUMERICAL_DATASETS + TEST_TRAIN_DATASETS

if __name__ == '__main__':
    results = defaultdict(lambda: np.zeros((len(TOPKS), len(DEPTHS))))

    for i in range(len(DEPTHS) * len(TOPKS) * len(DATASETS)):
        with open(f'topk/results/speed_new/{i}.out', 'r') as fp:
            header = fp.readline()
            if header == '':
                continue

            dataset, k, depth = header.split(' ')
            k_idx = TOPKS.index(int(k))
            depth_idx = DEPTHS.index(int(depth))
            speed = fp.readline()
            if speed != '':
                results[dataset][k_idx][depth_idx] = float(speed)

    for dataset in DATASETS:
        plt.figure()
        for k_idx, topk in enumerate(TOPKS):
            speeds = results[dataset][k_idx]
            speeds = speeds[speeds != 0] # remove failed runs
            plt.plot(DEPTHS[:len(speeds)], speeds, label=f'k={topk}', marker='o')
        plt.title(f'{dataset}')
        plt.xlabel('depth')
        plt.ylabel('speed (s)')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'./topk/results/figs/speed/{dataset}.pdf', format='pdf', bbox_inches='tight')
        plt.clf()