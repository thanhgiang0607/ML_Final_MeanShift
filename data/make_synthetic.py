from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

N = 1500
SEEDS = range(10)
OUT = Path(__file__).parent / "synthetic"


def blobs(seed):
    return make_blobs(n_samples=N, centers=4, cluster_std=0.6, random_state=seed)


def blobs_varied(seed):
    return make_blobs(n_samples=N, centers=3, cluster_std=[0.5, 1.5, 2.5], random_state=seed)


def blobs_aniso(seed):
    X, y = make_blobs(n_samples=N, centers=3, random_state=seed)
    return X @ np.array([[0.6, -0.6], [-0.4, 0.8]]), y


def moons(seed):
    return make_moons(n_samples=N, noise=0.05, random_state=seed)


def circles(seed):
    return make_circles(n_samples=N, factor=0.5, noise=0.05, random_state=seed)


DATASETS = {
    "blobs": blobs,
    "blobs_varied": blobs_varied,
    "blobs_aniso": blobs_aniso,
    "moons": moons,
    "circles": circles,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in DATASETS.items():
        for seed in SEEDS:
            X, y = make(seed)
            data = np.column_stack([X, y])
            np.savetxt(OUT / f"{name}_seed{seed}.csv", data, delimiter=",",
                       header="x,y,label", comments="", fmt=["%.6f", "%.6f", "%d"])
    print(f"Đã lưu {len(DATASETS) * len(SEEDS)} file vào {OUT}")
