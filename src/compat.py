import pickle

import numpy as np


def patch_numpy_legacy_pickle_aliases() -> None:
    if not hasattr(np, "loads"):
        np.loads = pickle.loads
