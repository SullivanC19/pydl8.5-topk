import subprocess
import os

import time
from typing import Dict, Any

import numpy as np

MURTREE_EXECUTABLE = os.path.join("topk", "algorithms", "murtree", "LinuxRelease", "MurTree")
TMP_SOLU_FILE = "_tmp_solu_{id}"
TMP_DATA_FILE = "_tmp_file_{id}"

def search(
        X_train,
        y_train,
        max_depth: int = 4,
        regularization: float = 0.0,
        time_limit: int = 10,
) -> Dict[str, Any]:
    assert(((X_train == 0) | (X_train == 1)).all())

    id = np.random.randint(0, 1000000)
    tmp_data_file = TMP_DATA_FILE.format(id=id)
    tmp_solu_file = TMP_SOLU_FILE.format(id=id)

    # set up temporary data file for murtree to read
    A = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
    A[:, 1:] = X_train
    A[:, 0] = y_train
    np.savetxt(tmp_data_file, A, fmt='%d')

    start = time.perf_counter()
    subprocess.run(
        [
            MURTREE_EXECUTABLE,
            "-file", tmp_data_file,
            "-result-file", tmp_solu_file,
            "-max-depth", str(max_depth),
            "-sparse-coefficient", str(regularization),
            "-time", str(time_limit),
            "-verbose", "true",
        ]
    )
    end = time.perf_counter()

    with open(tmp_solu_file, 'r') as fp:
        timeout = int(fp.readlines()[3]) == -1

    os.remove(tmp_solu_file)
    os.remove(tmp_data_file)

    return {
        'tree': None,
        'time': end - start,
        'timeout': timeout,
    }