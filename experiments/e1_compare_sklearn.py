from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KernelDensity
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom
from meanshift.bandwidth import estimate_bandwidth
from common import DATASETS, SUBSAMPLE, load_real, out_dir

OUT = out_dir("e1")
ALPHAS = [0.25, 0.5, 1.0]
SEARCHES = ["brute", "kdtree"]


def _center_metrics(a, b, bandwidth):
    na, nb = len(a), len(b)
    if not na or not nb:
        return {"matched_centers": 0, "unmatched_custom_centers": na,
                "unmatched_sklearn_centers": nb, "center_error_max": None,
                "center_error_mean": None, "center_error_max_over_h": None,
                "center_error_mean_over_h": None}
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    distances = cost[rows, cols]
    unmatched = abs(na - nb)
    return {"matched_centers": len(distances), "unmatched_custom_centers": max(0, na - nb),
            "unmatched_sklearn_centers": max(0, nb - na),
            "center_error_max": float(distances.max()) if len(distances) else None,
            "center_error_mean": float(distances.mean()) if len(distances) else None,
            "center_error_max_over_h": float(distances.max() / bandwidth) if len(distances) else None,
            "center_error_mean_over_h": float(distances.mean() / bandwidth) if len(distances) else None,
            "center_count_difference": unmatched,
            "center_error_max_penalized_over_h": float(max(distances.max() if len(distances) else 0.0, bandwidth if unmatched else 0.0) / bandwidth)}


def _timed_fit(factory, X):
    started = time.perf_counter()
    try:
        model = factory().fit(X)
    except ValueError as exc:
        return None, time.perf_counter() - started, str(exc)
    return model, time.perf_counter() - started, ""


def run(seeds=10):
    if seeds <= 0:
        raise ValueError("--seeds phải là số nguyên dương")
    rows, example_cache = [], {}
    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            for seed in range(seeds):
                X, truth = load_real(dataset, seed)
                base_h = estimate_bandwidth(X, quantile=0.3)
                if not np.isfinite(base_h) or base_h <= 0:
                    raise ValueError(f"Bandwidth ước lượng không hợp lệ ({dataset}, seed {seed}): {base_h}")
                for alpha in ALPHAS:
                    h = float(alpha * base_h)
                    for bin_seeding in (False, True):
                        reference, reference_elapsed, ref_error = _timed_fit(lambda: MeanShift(
                            bandwidth=h, bin_seeding=bin_seeding, cluster_all=True, n_jobs=1), X)
                        for search in SEARCHES:
                            model, elapsed, custom_error = _timed_fit(lambda: MeanShiftCustom(
                                bandwidth=h, bin_seeding=bin_seeding, neighbor_search=search, cluster_all=True), X)
                            base = {"dataset": dataset, "seed": seed, "alpha": alpha,
                                "base_bandwidth": base_h, "bandwidth": h,
                                "bin_seeding": bin_seeding, "neighbor_search": search,
                                "custom_fit_seconds": elapsed, "sklearn_fit_seconds": reference_elapsed,
                                "custom_error": custom_error, "sklearn_error": ref_error}
                            if model is None or reference is None:
                                both = model is None and reference is None
                                rows.append({**base, "outcome": "both_error" if both else "one_error",
                                    "cluster_count_equal": both, "labels_exact_equal": both})
                                continue
                            ari = float(adjusted_rand_score(model.labels_, reference.labels_))
                            centers = _center_metrics(model.cluster_centers_, reference.cluster_centers_, h)
                            row = {**base, "outcome": "ok",
                                "custom_n_clusters": len(model.cluster_centers_),
                                "sklearn_n_clusters": len(reference.cluster_centers_),
                                "cluster_count_equal": len(model.cluster_centers_) == len(reference.cluster_centers_),
                                "ari_custom_vs_sklearn": ari,
                                "labels_exact_equal": bool(np.array_equal(model.labels_, reference.labels_)),
                                "custom_n_iter": int(model.n_iter_), "sklearn_n_iter": int(reference.n_iter_),
                                "custom_truth_ari": float(adjusted_rand_score(truth, model.labels_)),
                                "sklearn_truth_ari": float(adjusted_rand_score(truth, reference.labels_)),
                                **centers}
                            rows.append(row)
                            if seed == 0 and alpha == 1.0 and not bin_seeding:
                                example_cache[(dataset, search)] = (X, truth, model, reference)
                pd.DataFrame(rows).to_csv(OUT / "e1_partial_runs.csv", index=False)
                print(f"[{dataset}] seed {seed + 1}/{seeds} complete", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "e1_runs.csv", index=False)
    keys = ["dataset", "alpha", "bin_seeding", "neighbor_search"]
    runs["both_error"] = runs["outcome"] == "both_error"
    summary = runs.groupby(keys, as_index=False).agg(
        runs=("seed", "count"), both_error_runs=("both_error", "sum"), cluster_count_equal_rate=("cluster_count_equal", "mean"),
        labels_exact_equal_rate=("labels_exact_equal", "mean"), ari_min=("ari_custom_vs_sklearn", "min"),
        ari_mean=("ari_custom_vs_sklearn", "mean"), center_error_max_over_h=("center_error_max_over_h", "max"),
        center_error_mean_over_h=("center_error_mean_over_h", "mean"),
        custom_truth_ari_mean=("custom_truth_ari", "mean"), sklearn_truth_ari_mean=("sklearn_truth_ari", "mean"),
        custom_seconds_mean=("custom_fit_seconds", "mean"), sklearn_seconds_mean=("sklearn_fit_seconds", "mean"))
    summary.to_csv(OUT / "e1_summary.csv", index=False)
    ok = runs["outcome"] == "ok"
    runs["comparison_mismatch"] = (runs["outcome"] == "one_error") | (ok & (
        ~runs["labels_exact_equal"].astype(bool) | ~runs["cluster_count_equal"].astype(bool) |
        (runs["center_error_max_over_h"].fillna(np.inf) > 1e-10) |
        (runs["custom_n_iter"] != runs["sklearn_n_iter"])))
    runs.to_csv(OUT / "e1_runs.csv", index=False)
    mismatch = runs.loc[runs["comparison_mismatch"]].copy()
    mismatch.to_csv(OUT / "e1_mismatches.csv", index=False)

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 5 * len(DATASETS)), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        X, truth, model, reference = example_cache[(dataset, "kdtree")]
        P = X
        cost = np.linalg.norm(model.cluster_centers_[:, None] - reference.cluster_centers_[None, :], axis=2)
        rr, cc = linear_sum_assignment(cost)
        mapping = dict(zip(rr, cc))
        for j, (estimator, title) in enumerate(((model, "custom/kdtree"), (reference, "sklearn"))):
            ax = axes[i, j]
            labels = estimator.labels_
            for k in np.unique(labels):
                color_index = mapping.get(k, len(reference.cluster_centers_) + k) if j == 0 else k
                ax.scatter(*P[labels == k].T, s=8, color=plt.get_cmap("tab20")(int(color_index) % 20), alpha=.7)
            ax.scatter(*estimator.cluster_centers_.T, marker="x", s=60, c="black", linewidths=1.5)
            ax.set_title(f"{dataset} — {title} (toàn bộ dữ liệu, α = 1, {len(estimator.cluster_centers_)} cụm)")
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "e1_examples.png", dpi=160)
    plt.close(fig)

    X, _ = load_real("iris")
    h_diag = 0.5 * estimate_bandwidth(X, quantile=0.3)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    density = np.exp(KernelDensity(kernel="gaussian", bandwidth=h_diag).fit(X).score_samples(grid)).reshape(gx.shape)
    all_paths = MeanShiftCustom(bandwidth=h_diag, record_trajectories=True).fit(X).trajectories_
    start = int(np.argmax([len(t) for t in all_paths]))
    diagnostic = MeanShiftCustom(bandwidth=h_diag, kernel="gaussian", gaussian_cutoff=3.0,
        seeds=X[[start]], record_trajectories=True).fit(X)
    trajectory = diagnostic.trajectories_[0]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.contourf(gx, gy, density, levels=24, cmap="magma", alpha=.78)
    ax.scatter(*X.T, s=5, c="white", alpha=.32, label="samples")
    ax.plot(trajectory[:, 0], trajectory[:, 1], "c.-", label="truncated Gaussian shift (cutoff=3h)")
    ax.set_title("Iris (PCA 2D): KDE Gauss đầy đủ và quỹ đạo Gauss cắt ở 3h")
    ax.legend(loc="best")
    fig.savefig(OUT / "e1_kde_trajectory.png", dpi=160)
    plt.close(fig)
    kde = KernelDensity(kernel="gaussian", bandwidth=h_diag).fit(X)
    trajectory_rows = []
    for kernel in ("flat", "gaussian"):
        fitted = MeanShiftCustom(bandwidth=h_diag, kernel=kernel, seeds=X[[start]], record_trajectories=True).fit(X)
        for step, point in enumerate(fitted.trajectories_[0]):
            u = np.sum((X - point) ** 2, axis=1) / h_diag**2
            full_gaussian = np.exp(-0.5 * u).mean() / (2 * np.pi * h_diag**2)
            epanechnikov = 2 * np.maximum(1 - u, 0).mean() / (np.pi * h_diag**2)
            reference_density = float(np.exp(kde.score_samples(point[None]))[0])
            trajectory_rows.append({"kernel": kernel, "start_index": start, "step": step, "x": point[0], "y": point[1],
                "gaussian_kde_direct": full_gaussian, "gaussian_kde_sklearn": reference_density,
                "epanechnikov_kde": epanechnikov})
    trajectory_frame = pd.DataFrame(trajectory_rows)
    trajectory_frame.to_csv(OUT / "e1_density_steps.csv", index=False)
    kde_error = float(np.max(np.abs(trajectory_frame.gaussian_kde_direct - trajectory_frame.gaussian_kde_sklearn)))
    np.testing.assert_allclose(trajectory_frame.gaussian_kde_direct, trajectory_frame.gaussian_kde_sklearn, rtol=1e-10, atol=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, kernel, column in zip(axes, ("flat", "gaussian"), ("epanechnikov_kde", "gaussian_kde_direct")):
        subset = trajectory_frame[trajectory_frame.kernel == kernel]
        ax.plot(subset.step, subset[column], "o-")
        ax.set(xlabel="Iteration", ylabel="KDE", title=f"{kernel} shift: {column}")
    fig.savefig(OUT / "e1_density_steps.png", dpi=160)
    plt.close(fig)
    (OUT / "e1_kde_check.json").write_text(json.dumps({"gaussian_kde_max_absolute_error": kde_error,
        "scope": "Direct full Gaussian KDE versus sklearn at recorded trajectory points; not a proof of truncated-update convergence."}, indent=2), encoding="utf-8")

    metadata = {"experiment": "E1 custom vs sklearn MeanShift", "seeds": list(range(seeds)),
        "datasets": DATASETS, "alphas": ALPHAS, "bin_seeding": [False, True],
        "neighbor_search": SEARCHES, "rows": len(runs),
        "data": f"sklearn.datasets (Iris, Wine, Breast Cancer), StandardScaler rồi PCA 2 thành phần; seed 0 = toàn bộ dữ liệu, seed ≥ 1 = mẫu con phân tầng {SUBSAMPLE:.0%}",
        "diagnostic_data": "Iris PCA 2D, toàn bộ dữ liệu (cho hình KDE và mật độ theo bước)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "config": {"bandwidth": "alpha * estimate_bandwidth(X, quantile=0.3)", "quantile": 0.3,
            "custom_kernel": "flat; uniform-ball/Epanechnikov KDE gradient weights",
            "diagnostic_kernel": "Gaussian cutoff at 3h; plotted sklearn Gaussian KDE has full support",
            "tol": 1e-3, "max_iter": 300, "merge_radius": "h", "cluster_all": True,
            "sklearn_n_jobs": 1, "threadpool_limit": 1, "reference_fits_reused_across_search": True,
            "time_note": "Each custom backend fit once; sklearn reference fit once and reused across backend comparisons"},
        "metric_caveats": {"center_error": "Hungarian matched-pair Euclidean errors normalized by h; unmatched center counts are reported separately and receive no arbitrary distance penalty.",
            "truth_ari": "ARI so với nhãn lớp thật; lớp thật không nhất thiết trùng với mode mật độ.",
            "timing": "Single-process wall-clock; sensitive to machine load.",
            "KDE": "Plotted surface is full-support Gaussian KDE; custom Gaussian trajectory truncates neighbors at 3h. Flat updates are uniform-ball/Epanechnikov gradient weights, not Gaussian KDE. No monotonic-density claim is made for truncated Gaussian updates."}}
    (OUT / "e1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cả hai cùng báo lỗi (không seed nào có láng giềng): {int(runs.both_error.sum())}/{len(runs)}; "
          f"chỉ một bên lỗi: {int((runs.outcome == 'one_error').sum())}")
    count_equal = runs["cluster_count_equal"].mean() * 100
    exact = runs["labels_exact_equal"].mean() * 100
    print(f"Cùng số cụm: {count_equal:.1f}%; nhãn giống hệt: {exact:.1f}%; ARI nhỏ nhất: {runs.ari_custom_vs_sklearn.min():.6f}; lệch tâm lớn nhất/h: {runs.center_error_max_over_h.max():.6f}")
    print(f"Số lượt khác ít nhất một tiêu chí: {len(mismatch)}; danh sách đầy đủ: {OUT / 'e1_mismatches.csv'}")
    if len(mismatch):
        print(mismatch[["dataset", "seed", "alpha", "bin_seeding", "neighbor_search", "custom_n_clusters", "sklearn_n_clusters", "ari_custom_vs_sklearn", "center_error_max_over_h"]].to_string(index=False))
    print(f"Đã lưu {len(runs)} lượt chạy tại {OUT}")
    return runs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="Số seed đầu tiên chạy (mặc định 10).")
    args = parser.parse_args()
    run(args.seeds)


if __name__ == "__main__":
    main()
