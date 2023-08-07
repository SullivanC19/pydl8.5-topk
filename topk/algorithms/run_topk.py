import time
from typing import Dict, Any
from pydl85 import DL85Classifier, Cache_Type
import numpy as np

from .binary_classification_tree import BinaryClassificationTree


# return the error with an added per-leaf sparsity penalty
def sparse_misclassification_error(sup_iter, sparsity_penalty):
    supports = list(sup_iter)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex] + sparsity_penalty, maxindex


def search(
        X_train,
        y_train,
        max_depth: int = 3,
        k: int = 4,
        regularization: float = 0.0,
        time_limit: int = 0,
) -> Dict[str, Any]:
    assert(((X_train == 0) | (X_train == 1)).all())

    parameters = {
        "k": k,
        "max_depth": max_depth,
        "depth_two_special_algo": False,
        "similar_lb": False,
        "similar_for_branching": False,
        "desc": True,
        "repeat_sort": True,
        "cache_type": Cache_Type.Cache_HashCover,
        "time_limit": time_limit,
    }

    if regularization:
        sparsity_penalty = regularization * X_train.shape[0]
        parameters["fast_error_function"] = lambda sup_iter: sparse_misclassification_error(sup_iter, sparsity_penalty)

    start = time.perf_counter()
    clf = DL85Classifier(**parameters)
    clf.fit(X_train, y_train)
    end = time.perf_counter()

    tree = parse(clf)
    tree.fit(X_train, y_train, num_labels=max(y_train) + 1)
    pred = clf.predict(X_train)
    pred_parsed = tree.predict(X_train)

    return {
        'tree': tree,
        'time': end - start,
        'timeout': clf.timeout_,
    }

def parse(clf: DL85Classifier) -> BinaryClassificationTree:
    def parse_node(node: dict) -> BinaryClassificationTree:
        if "value" in node:
            return BinaryClassificationTree()
        return BinaryClassificationTree(
            parse_node(node["right"]),
            parse_node(node["left"]),
            int(node["feat"]))
    return parse_node(clf.base_tree_)
