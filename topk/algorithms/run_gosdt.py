import time
from typing import Dict, Any
import pandas as pd
from multiprocessing import Process, Queue
from gosdt import GOSDT

from .decorators import silence
from .binary_classification_tree import BinaryClassificationTree

def search_target(config, X_train, y_train, queue: Queue):
    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train)

    start = time.perf_counter()
    with silence():
        clf = GOSDT(config)
        clf.fit(df_X_train, df_y_train)
    end = time.perf_counter()

    tree = parse(clf)
    tree.fit(X_train, y_train, num_labels=max(y_train) + 1)

    queue.put({
        'tree': tree,
        'time': end - start,
        'timeout': clf.timeout
    })


def search(
        X_train,
        y_train,
        max_depth: int = 0,
        regularization: float = 0.01,
        time_limit: int = 10,
) -> Dict[str, Any]:
    assert(((X_train == 0) | (X_train == 1)).all())

    config = {
        'depth_budget': max_depth,
        'regularization': regularization,
        'time_limit': time_limit,
        'allow_small_reg': True,
    }

    queue = Queue()
    p = Process(target=search_target, args=(config, X_train, y_train, queue))
    p.start()
    p.join()
    return queue.get()

def parse(clf: GOSDT) -> BinaryClassificationTree:
    def parse_node(node: dict) -> BinaryClassificationTree:
        if "prediction" in node:
            return BinaryClassificationTree()
        assert(node["relation"] == "==")
        assert(node["reference"] == 1.0)
        return BinaryClassificationTree(
            parse_node(node["false"]),
            parse_node(node["true"]),
            int(node["feature"]))

    root = clf.tree.source
    return parse_node(root)