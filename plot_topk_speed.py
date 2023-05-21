import matplotlib.pyplot as plt

DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TOPKS = [1, 2, 4, 8, 12, 16]
DATASETS = ['connect-4', 'artificial-characters']

if __name__ == '__main__':
    for dataset in DATASETS:
        plt.figure()
        for topk in TOPKS:
            speeds = []
            for depth in DEPTHS:
                with open(f'./{dataset}_k={topk}_depth={depth}.out', 'r') as fp:
                    speed = float(fp.read())
                    speeds.append(speed)
            plt.plot(DEPTHS, speed, label=f'k={topk}', marker='o')
        plt.title(f'{dataset}')
        plt.xlabel('depth')
        plt.ylabel('speed (s)')
        plt.legend()
        plt.savefig(f'./{dataset}/topk/results/figs/speed/{topk}/speed.pdf', format='pdf', bbox_inches='tight')
        plt.clf()