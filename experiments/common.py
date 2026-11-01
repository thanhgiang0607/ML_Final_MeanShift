import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = ["iris", "wine", "breast_cancer"]
LOADERS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
TRUE_K = {"iris": 3, "wine": 3, "breast_cancer": 2}
SUBSAMPLE = 0.8


def load_raw(name):
    data = LOADERS[name]()
    return data.data.astype(np.float64), np.asarray(data.target)


def load_real(name, seed=0):
    X, y = load_raw(name)
    if seed > 0:
        X, _, y, _ = train_test_split(X, y, train_size=SUBSAMPLE, stratify=y, random_state=seed)
    return StandardScaler().fit_transform(X), y


def pca_2d(X):
    return PCA(n_components=2, random_state=0).fit(X)


def load_real_2d(name):
    X, y = load_real(name)
    return pca_2d(X).transform(X), y


def timeit(fn, repeat=3):
    times, result = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def out_dir(name):
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
