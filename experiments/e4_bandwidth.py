import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scipy
import sklearn

from common import DATASETS, ROOT, load_synthetic, synthetic_path
from meanshift.bandwidth import estimate_bandwidth
from meanshift.core import MeanShiftCustom


ALPHAS = np.geomspace(0.1, 3.0, 15)
DEFAULT_SEEDS = list(range(10))
QUANTILE = 0.3
SAFE_RATIO = 0.9


def load_scaled(dataset, seed):
    X, y = load_synthetic(dataset, seed)
    X = StandardScaler().fit_transform(X)
    return X, y


def run_one(dataset, seed, alpha):
    X, y_true = load_scaled(dataset, seed)
    h0 = estimate_bandwidth(X, quantile=QUANTILE)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError(
            f"Bandwidth mặc định không hợp lệ: dataset={dataset}, "
            f"seed={seed}, h0={h0}"
        )

    h = float(alpha * h0)
    elapsed_times = []
    labels = None

    for _ in range(3):
        model = MeanShiftCustom(
            bandwidth=h,
            bin_seeding=True,
            neighbor_search="kdtree",
        )

        start = time.perf_counter()
        model.fit(X)
        elapsed_times.append(time.perf_counter() - start)

        labels = model.labels_

    y_pred = np.asarray(labels)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

    return {
        "dataset": dataset,
        "seed": seed,
        "alpha": float(alpha),
        "h0": float(h0),
        "bandwidth": h,
        "n_clusters": int(n_clusters),
        "n_true_clusters": int(len(np.unique(y_true))),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "time_seconds": float(np.median(elapsed_times)),
    }


def summarize_results(runs):
    summary = (
        runs.groupby(["dataset", "alpha"], as_index=False)
        .agg(
            n_true_clusters=("n_true_clusters", "first"),
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_std=("n_clusters", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            time_mean=("time_seconds", "mean"),
            time_std=("time_seconds", "std"),
        )
    )
    std_columns = [col for col in summary.columns if col.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)

    return summary


def alpha_table(summary):
    rows = []
    for dataset in DATASETS:
        data = summary[summary["dataset"] == dataset].sort_values("alpha")
        best = data.loc[data["ari_mean"].idxmax()]
        default = data.loc[(data["alpha"] - 1.0).abs().idxmin()]
        safe = data[data["ari_mean"] >= SAFE_RATIO * best["ari_mean"]]
        rows.append({
            "dataset": dataset,
            "n_true_clusters": int(best["n_true_clusters"]),
            "best_alpha": float(best["alpha"]),
            "best_ari": float(best["ari_mean"]),
            "best_n_clusters": float(best["n_clusters_mean"]),
            "default_ari": float(default["ari_mean"]),
            "default_n_clusters": float(default["n_clusters_mean"]),
            "ari_gap": float(best["ari_mean"] - default["ari_mean"]),
            "safe_alpha_min": float(safe["alpha"].min()),
            "safe_alpha_max": float(safe["alpha"].max()),
            "safe_n_alphas": int(len(safe)),
            "default_in_safe": bool((safe["alpha"] == default["alpha"]).any()),
        })
    return pd.DataFrame(rows)


def plot_results(summary, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, dataset in enumerate(DATASETS):
        color = f"C{i}"
        data = summary[summary["dataset"] == dataset].sort_values("alpha")

        x = data["alpha"].to_numpy()
        y_clusters = data["n_clusters_mean"].to_numpy()
        err_clusters = data["n_clusters_std"].to_numpy()

        axes[0].plot(x, y_clusters, marker="o", color=color, label=dataset)
        axes[0].fill_between(
            x,
            np.maximum(y_clusters - err_clusters, 1),
            y_clusters + err_clusters,
            color=color,
            alpha=0.15,
        )
        axes[0].axhline(
            data["n_true_clusters"].iloc[0],
            color=color,
            linestyle=":",
            linewidth=1,
        )
        y_ari = data["ari_mean"].to_numpy()
        err_ari = data["ari_std"].to_numpy()

        axes[1].plot(x, y_ari, marker="o", color=color, label=dataset)
        axes[1].fill_between(
            x,
            np.clip(y_ari - err_ari, 0, 1),
            np.clip(y_ari + err_ari, 0, 1),
            color=color,
            alpha=0.15,
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            x=1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="alpha = 1 (h mặc định)",
        )
        ax.set_xlabel("Alpha (h = alpha × h0)")
        ax.grid(True, alpha=0.3)

    axes[0].set_yscale("log")
    axes[0].set_title("Độ nhạy số lượng cụm theo bandwidth\n(đường chấm: số cụm thật)")
    axes[0].set_ylabel("Số cụm (thang log)")
    axes[1].set_title("Độ nhạy ARI theo bandwidth")
    axes[1].set_ylabel("ARI")
    axes[1].set_ylim(-0.02, 1.02)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="E4: Khảo sát độ nhạy của Mean Shift theo bandwidth."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Số seed chạy cho mỗi dataset (mặc định: 10).",
    )
    args = parser.parse_args()

    if args.seeds < 1 or args.seeds > 10:
        parser.error("--seeds phải nằm trong khoảng từ 1 đến 10.")

    seeds = DEFAULT_SEEDS[:args.seeds]

    output_dir = ROOT / "results" / "e4"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    alphas_to_run = sorted(set(ALPHAS.tolist() + [1.0]))

    total = len(DATASETS) * len(seeds) * len(alphas_to_run)
    current = 0

    for dataset in DATASETS:
        for seed in seeds:
            for alpha in alphas_to_run:
                current += 1
                print(
                    f"[{current}/{total}] "
                    f"dataset={dataset}, seed={seed}, alpha={alpha:.4f}"
                )

                result = run_one(dataset, seed, alpha)
                all_runs.append(result)

    runs = pd.DataFrame(all_runs)
    summary = summarize_results(runs)
    table = alpha_table(summary)

    runs_path = output_dir / "e4_synthetic_runs.csv"
    summary_path = output_dir / "e4_synthetic_summary.csv"
    table_path = output_dir / "e4_synthetic_alpha_summary.csv"
    plot_path = output_dir / "e4_synthetic.png"
    metadata_path = output_dir / "e4_metadata.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    table.to_csv(table_path, index=False)
    plot_results(summary, plot_path)

    used = [synthetic_path(d, s) for d in DATASETS for s in seeds]
    metadata = {
        "experiment": "E4 – độ nhạy theo bandwidth (phần 1: dữ liệu giả lập)",
        "datasets": DATASETS,
        "seeds": seeds,
        "alphas": alphas_to_run,
        "bandwidth": f"h = alpha × h0, h0 = meanshift.estimate_bandwidth(X, quantile={QUANTILE})",
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "standardization": "StandardScaler riêng cho từng (bộ, seed)",
        "timing": "time.perf_counter quanh fit, 3 lần, lấy trung vị",
        "best_alpha": "alpha có ARI trung bình cao nhất trên chính các seed đã chạy (cận trên, dùng nhãn thật)",
        "safe_range": f"các alpha có ARI trung bình >= {SAFE_RATIO} × ARI tốt nhất của bộ đó",
        "input_sha256": {
            str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in used
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== TÓM TẮT E4 =====")
    print(f"Số lần chạy: {len(runs)}")
    print(f"Kết quả chi tiết: {runs_path}")
    print(f"Kết quả tổng hợp: {summary_path}")
    print(f"Bảng alpha theo từng bộ: {table_path}")
    print(f"Biểu đồ: {plot_path}")
    print()
    print(table[[
        "dataset", "best_alpha", "best_ari", "default_ari",
        "safe_alpha_min", "safe_alpha_max", "default_in_safe",
    ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
