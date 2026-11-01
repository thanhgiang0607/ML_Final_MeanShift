import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"

DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]
TRUE_K = {"blobs": 4, "blobs_varied": 3, "blobs_aniso": 3, "moons": 2, "circles": 2}


def synthetic_path(name, seed):
    return DATA / f"{name}_seed{seed}.csv"


def load_synthetic(name, seed):
    path = synthetic_path(name, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python3 data/make_synthetic.py")
    frame = pd.read_csv(path)
    if not {"x", "y", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"CSV phải có các cột x, y, label và ít nhất 1 dòng: {path}")
    X = frame[["x", "y"]].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy()
    if not np.isfinite(X).all():
        raise ValueError(f"Dữ liệu có giá trị không hữu hạn: {path}")
    return X, y


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
