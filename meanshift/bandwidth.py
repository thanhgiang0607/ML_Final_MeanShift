import numpy as np
from scipy.spatial import cKDTree


def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0):
    if not 0 <= quantile <= 1:
        raise ValueError(f"quantile phải nằm trong [0, 1], nhận được {quantile!r}")

    X = np.asarray(X, dtype=float)

    if n_samples is not None:
        if n_samples < 1:
            raise ValueError(f"n_samples phải >= 1 hoặc None, nhận được {n_samples!r}")
        rng = np.random.RandomState(random_state)
        X = X[rng.permutation(X.shape[0])[:n_samples]]

    k = max(1, int(X.shape[0] * quantile))

    dist, _ = cKDTree(X).query(X, k=k)
    kth = dist if k == 1 else dist[:, -1]
    return float(np.mean(kth))
