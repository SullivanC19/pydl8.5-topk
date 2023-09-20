from topk.algorithms.run_topk import search as topk_search

DEPTH_VALUES = list(range(1, 9))
TIME_LIMIT = 600

MODELS = {
    "Top-1": lambda X, y, depth: topk_search(X, y, k=1, max_depth=depth, time_limit=TIME_LIMIT),
    "Top-2": lambda X, y, depth: topk_search(X, y, k=2, max_depth=depth, time_limit=TIME_LIMIT),
    "Top-4": lambda X, y, depth: topk_search(X, y, k=4, max_depth=depth, time_limit=TIME_LIMIT),
    "Top-8": lambda X, y, depth: topk_search(X, y, k=8, max_depth=depth, time_limit=TIME_LIMIT),
    "Top-12": lambda X, y, depth: topk_search(X, y, k=8, max_depth=depth, time_limit=TIME_LIMIT),
    "Top-16": lambda X, y, depth: topk_search(X, y, k=16, max_depth=depth, time_limit=TIME_LIMIT),
}