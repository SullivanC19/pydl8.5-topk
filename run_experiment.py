from argparse import ArgumentParser

from topk.globals import EXPERIMENTS, ALL_DATASETS
from topk.experiments.fig2.runner import main as run_fig2
from topk.experiments.fig3.runner import main as run_fig3
from topk.experiments.fig4.runner import main as run_fig4


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d', default=None, type=str)
    parser.add_argument('--dataset_index', '-di', default=None, type=int)
    parser.add_argument('--experiment', '-e', default=None, type=str)
    parser.add_argument('--experiment_index', '-ei', default=None, type=int)
    parser.add_argument('--job_index', '-ji', default=None, type=int)

    args = parser.parse_args()

    if args.job_index is not None:
        args.dataset_index = int(args.job_index / len(EXPERIMENTS))
        args.experiment_index = int(args.job_index % len(EXPERIMENTS))

    assert args.dataset is not None or args.dataset_index is not None
    assert args.experiment is not None or args.experiment_index is not None

    dataset = args.dataset if args.dataset_index is None else ALL_DATASETS[args.dataset_index]
    experiment = args.experiment if args.experiment_index is None else EXPERIMENTS[args.experiment_index]

    assert dataset in ALL_DATASETS
    assert experiment in EXPERIMENTS

    experiment_runner = None
    if experiment == "fig2":
        run_fig2(dataset)
    if experiment == "fig3":
        run_fig3(dataset)
    if experiment == "fig4":
        run_fig4(dataset)