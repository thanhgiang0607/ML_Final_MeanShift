import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import MeanShift
from sklearn.cluster import get_bin_seeds as sk_get_bin_seeds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom, estimate_bandwidth
from common import DATASETS, load_raw, load_real, out_dir

OUT = out_dir("e3")
METHODS = ["custom brute", "custom kdtree", "custom kdtree + bin", "sklearn", "sklearn + bin"]
STYLE = {"custom brute": ("C0", "o-"), "custom kdtree": ("C1", "s-"), "custom kdtree + bin": ("C2", "^-"),
         "sklearn": ("C3", "D--"), "sklearn + bin": ("C4", "v--")}
N_SWEEP_DATASET = "breast_cancer"
N_VALUES = [100, 200, 300, 400, 500, 569]
ALPHAS = [0.25, 0.5, 1.0, 2.0]
REPEATS = 3


def make_model(method, h):
    if method == "custom brute":
        return MeanShiftCustom(bandwidth=h, neighbor_search="brute")
    if method == "custom kdtree":
        return MeanShiftCustom(bandwidth=h, neighbor_search="kdtree")
    if method == "custom kdtree + bin":
        return MeanShiftCustom(bandwidth=h, neighbor_search="kdtree", bin_seeding=True)
    if method == "sklearn":
        return MeanShift(bandwidth=h, n_jobs=1)
    if method == "sklearn + bin":
        return MeanShift(bandwidth=h, bin_seeding=True, n_jobs=1)
    raise ValueError(method)


def n_seeds(method, X, h):
    if method.endswith("+ bin"):
        return len(sk_get_bin_seeds(X, h, 1))
    return len(X)


def measure(method, X, h, repeats):
    try:
        make_model(method, h).fit(X)
    except ValueError as exc:
        return {"time_median_s": np.nan, "time_min_s": np.nan, "time_max_s": np.nan, "peak_memory_mb": np.nan,
                "n_seeds": n_seeds(method, X, h), "n_clusters": np.nan, "n_iter": np.nan, "error": str(exc)}
    times, model = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        model = make_model(method, h).fit(X)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    make_model(method, h).fit(X)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"time_median_s": float(np.median(times)), "time_min_s": float(np.min(times)),
            "time_max_s": float(np.max(times)), "peak_memory_mb": peak / 2**20,
            "n_seeds": n_seeds(method, X, h), "n_clusters": len(model.cluster_centers_),
            "n_iter": int(model.n_iter_), "error": ""}


def n_sweep_data():
    X_raw, y = load_raw(N_SWEEP_DATASET)
    X = StandardScaler().fit_transform(X_raw)
    h = estimate_bandwidth(X)
    for n in N_VALUES:
        if n >= len(X):
            yield n, X, h
        else:
            Xs, _ = train_test_split(X, train_size=n, stratify=y, random_state=0)
            yield n, Xs, h


def plot_vs_n(summary, column, ylabel, title, path, log_y=True):
    data = summary[summary.experiment == "n"]
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for method in METHODS:
        sub = data[data.method == method].sort_values("n")
        color, style = STYLE[method]
        ax.plot(sub.n, sub[column], style, color=color, label=method, ms=5)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
    ax.set_xlabel("Số điểm n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_h(summary, path):
    data = summary[summary.experiment == "h"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.3), constrained_layout=True)
    for ax, name in zip(axes, DATASETS):
        sub_ds = data[data.dataset == name]
        for method in METHODS:
            sub = sub_ds[sub_ds.method == method].sort_values("alpha")
            color, style = STYLE[method]
            ax.plot(sub.alpha, sub.time_median_s, style, color=color, label=method, ms=5)
        first = sub_ds.iloc[0]
        ax.set_title(f"{name} (n = {first.n}, d = {first.d})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ALPHAS, [str(a) for a in ALPHAS])
        ax.set_xlabel("α (h = α·h₀)")
        ax.set_ylabel("Thời gian trung vị (s, log)")
        ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Thời gian chạy theo bandwidth trên dữ liệu thật")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E3: thời gian chạy và bộ nhớ trên dữ liệu thật.")
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"số lần đo mỗi cấu hình (mặc định {REPEATS})")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats phải là số nguyên dương")

    rows = []
    with threadpool_limits(limits=1):
        for n, X, h in n_sweep_data():
            for method in METHODS:
                rows.append({"experiment": "n", "dataset": N_SWEEP_DATASET, "n": len(X), "d": X.shape[1],
                             "alpha": 1.0, "h": h, "method": method, **measure(method, X, h, args.repeats)})
            print(f"[n] n = {n} xong", flush=True)
        for name in DATASETS:
            X, _ = load_real(name)
            h0 = estimate_bandwidth(X)
            for alpha in ALPHAS:
                for method in METHODS:
                    rows.append({"experiment": "h", "dataset": name, "n": len(X), "d": X.shape[1],
                                 "alpha": alpha, "h": alpha * h0, "method": method,
                                 **measure(method, X, alpha * h0, args.repeats)})
            print(f"[h] {name} xong", flush=True)

    summary = pd.DataFrame(rows)
    brute = summary[summary.method == "custom brute"].set_index(["experiment", "dataset", "n", "alpha"]).time_median_s
    summary["speedup_vs_brute"] = [brute[(r.experiment, r.dataset, r.n, r.alpha)] / r.time_median_s
                                   for r in summary.itertuples()]
    summary.to_csv(OUT / "e3_summary.csv", index=False)

    plot_vs_n(summary, "time_median_s", "Thời gian trung vị (s, log)",
              f"Thời gian chạy theo n ({N_SWEEP_DATASET}, h cố định)", OUT / "e3_time_vs_n.png")
    plot_vs_n(summary, "peak_memory_mb", "Bộ nhớ đỉnh (MB)",
              f"Bộ nhớ đỉnh theo n ({N_SWEEP_DATASET}, h cố định)", OUT / "e3_memory_vs_n.png", log_y=False)
    plot_vs_h(summary, OUT / "e3_time_vs_h.png")

    env = (f"python {platform.python_version()}\nplatform {platform.platform()}\nprocessor {platform.processor()}\n"
           f"numpy {np.__version__}\nscipy {scipy.__version__}\nscikit-learn {sklearn.__version__}\n")
    (OUT / "env.txt").write_text(env, encoding="utf-8")
    metadata = {
        "experiment": "E3 – thời gian chạy và bộ nhớ trên dữ liệu thật",
        "n_sweep": {"dataset": N_SWEEP_DATASET, "n_values": N_VALUES,
                    "sampling": "chuẩn hoá toàn bộ dữ liệu rồi lấy mẫu con phân tầng random_state=0",
                    "h": "estimate_bandwidth trên toàn bộ dữ liệu chuẩn hoá, giữ cố định"},
        "h_sweep": {"datasets": DATASETS, "alphas": ALPHAS, "h": "α·estimate_bandwidth(X, quantile=0.3)"},
        "methods": METHODS, "repeats": args.repeats,
        "time": "time.perf_counter, trung vị các lần đo", "memory": "tracemalloc, đo ở 1 lần chạy riêng",
        "threadpool_limit": 1, "sklearn_n_jobs": 1,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (OUT / "e3_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nTheo n:")
    print(summary[summary.experiment == "n"].pivot(index="n", columns="method", values="time_median_s")
          [METHODS].round(4).to_string())
    print("\nTheo h:")
    print(summary[summary.experiment == "h"].pivot_table(index=["dataset", "alpha"], columns="method",
          values="time_median_s")[METHODS].round(4).to_string())
    print(f"\nĐã lưu kết quả E3 tại {OUT}")


if __name__ == "__main__":
    main()
