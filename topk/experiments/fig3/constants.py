from topk.algorithms.run_topk import search as topk_search
from topk.algorithms.run_gosdt import search as gosdt_search
from topk.algorithms.run_murtree import search as murtree_search

DEPTH_VALUES = [4, 5, 6]
TIME_LIMIT = 10
REGULARIZATION = 0.001
RUNS_PER_DATASET = 30

MODELS = {
    "GOSDT": lambda X, y, depth: gosdt_search(X, y, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "MurTree": lambda X, y, depth: murtree_search(X, y, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "Top-1": lambda X, y, depth: topk_search(X, y, k=1, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "Top-2": lambda X, y, depth: topk_search(X, y, k=2, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "Top-4": lambda X, y, depth: topk_search(X, y, k=4, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "Top-8": lambda X, y, depth: topk_search(X, y, k=8, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
    "Top-16": lambda X, y, depth: topk_search(X, y, k=16, max_depth=depth, time_limit=TIME_LIMIT, regularization=REGULARIZATION),
}