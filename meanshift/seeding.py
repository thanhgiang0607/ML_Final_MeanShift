import warnings

import numpy as np


def get_bin_seeds(X, bin_size, min_bin_freq=1):

    X = np.asarray(X)
    if bin_size == 0:
        return X

    cells, counts = np.unique(np.round(X / bin_size), axis=0, return_counts=True)

    seeds = cells[counts >= min_bin_freq].astype(np.float32)

    if len(seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds."
            % bin_size
        )
        return X
    return seeds * bin_size
