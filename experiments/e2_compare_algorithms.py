import argparse
import json
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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom, estimate_bandwidth
from common import DATASETS, SUBSAMPLE, TRUE_K, load_real, out_dir

OUT = out_dir("e2")
METHODS = ["MeanShift (h mặc định)", "MeanShift (h chọn)", "KMeans", "DBSCAN", "GMM"]
ALPHA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
EPS_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MIN_SAMPLES = 5
MIN_CORE_FRACTION = 0.5


def make_model(method, name, seed, params, h0):
    if method == "MeanShift (h mặc định)":
        return MeanShiftCustom(bandwidth=h0, bin_seeding=True)
    if method == "MeanShift (h chọn)":
        return MeanShiftCustom(bandwidth=params["alpha"] * h0, bin_seeding=True)
    if method == "KMeans":
        return KMeans(n_clusters=TRUE_K[name], n_init=10, random_state=seed)
    if method == "DBSCAN":
        return DBSCAN(eps=params["eps"], min_samples=MIN_SAMPLES)
    if method == "GMM":
        return GaussianMixture(n_components=TRUE_K[name], covariance_type="full", n_init=5, random_state=seed)
    raise ValueError(method)


def fit_labels(model, X):
    start = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - start
    labels = model.predict(X) if isinstance(model, GaussianMixture) else model.labels_
    return np.asarray(labels), elapsed


def unsupervised_score(X, labels, min_core=MIN_CORE_FRACTION):
    keep = labels != -1
    k = len(set(labels[keep].tolist()))
    if keep.mean() < min_core or k < 2 or k >= keep.sum():
        return -np.inf
    return float(silhouette_score(X[keep], labels[keep]))


def select_params(name):
    X, y = load_real(name, 0)
    h0 = estimate_bandwidth(X)
    grid = []
    for alpha in ALPHA_GRID:
        try:
            labels = MeanShiftCustom(bandwidth=alpha * h0, bin_seeding=True).fit_predict(X)
        except ValueError:
            grid.append({"dataset": name, "method": "MeanShift (h chọn)", "param": "alpha", "value": alpha,
                         "silhouette": -np.inf, "n_clusters": np.nan, "ari_for_reference": np.nan})
            continue
        grid.append({"dataset": name, "method": "MeanShift (h chọn)", "param": "alpha", "value": alpha,
                     "silhouette": unsupervised_score(X, labels), "n_clusters": len(set(labels.tolist())),
                     "ari_for_reference": adjusted_rand_score(y, labels)})
    for eps in EPS_GRID:
        labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X).labels_
        grid.append({"dataset": name, "method": "DBSCAN", "param": "eps", "value": eps,
                     "silhouette": unsupervised_score(X, labels),
                     "silhouette_relaxed": unsupervised_score(X, labels, min_core=0.0),
                     "noise_fraction": float(np.mean(labels == -1)),
                     "n_clusters": len(set(labels.tolist()) - {-1}),
                     "ari_for_reference": adjusted_rand_score(y, labels)})
    frame = pd.DataFrame(grid)
    rules = {}
    best = {}
    for m in ("MeanShift (h chọn)", "DBSCAN"):
        sub = frame[frame.method == m]
        column = "silhouette"
        if np.isinf(sub[column]).all() and "silhouette_relaxed" in sub and np.isfinite(sub.silhouette_relaxed).any():
            column = "silhouette_relaxed"
        rules[m] = "chặt" if column == "silhouette" else "nới (bỏ điều kiện tỉ lệ nhiễu)"
        best[m] = sub.sort_values([column, "value"], ascending=[False, True]).iloc[0]
        best[m] = best[m].copy()
        best[m]["score"] = best[m][column]
    ms, db = best["MeanShift (h chọn)"], best["DBSCAN"]
    return {"alpha": float(ms.value), "alpha_silhouette_seed0": float(ms.score), "alpha_rule": rules["MeanShift (h chọn)"],
            "eps": float(db.value), "eps_silhouette_seed0": float(db.score), "eps_rule": rules["DBSCAN"]}, frame


def main():
    parser = argparse.ArgumentParser(description="E2: so sánh Mean-Shift với K-Means, DBSCAN, GMM trên dữ liệu thật.")
    parser.add_argument("--seeds", type=int, default=10, help="seed 0 = toàn bộ dữ liệu, seed ≥ 1 = mẫu con (mặc định 10)")
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds phải là số nguyên dương")

    rows, param_rows, examples = [], [], {}
    with threadpool_limits(limits=1):
        params, grids = {}, []
        for name in DATASETS:
            params[name], grid = select_params(name)
            grids.append(grid)
            p = params[name]
            param_rows += [
                {"dataset": name, "method": "MeanShift (h mặc định)", "param": "alpha", "value": 1.0, "silhouette_seed0": np.nan},
                {"dataset": name, "method": "MeanShift (h chọn)", "param": "alpha", "value": p["alpha"], "silhouette_seed0": p["alpha_silhouette_seed0"]},
                {"dataset": name, "method": "KMeans", "param": "k", "value": TRUE_K[name], "silhouette_seed0": np.nan},
                {"dataset": name, "method": "DBSCAN", "param": "eps", "value": p["eps"], "silhouette_seed0": p["eps_silhouette_seed0"]},
                {"dataset": name, "method": "GMM", "param": "k", "value": TRUE_K[name], "silhouette_seed0": np.nan},
            ]
            print(f"[chọn tham số] {name}: α = {p['alpha']} ({p['alpha_rule']}), eps = {p['eps']} ({p['eps_rule']})", flush=True)

        for name in DATASETS:
            for seed in range(args.seeds):
                X, y = load_real(name, seed)
                h0 = estimate_bandwidth(X)
                for method in METHODS:
                    labels, elapsed = fit_labels(make_model(method, name, seed, params[name], h0), X)
                    if seed == 0:
                        examples[(name, method)] = (X, labels)
                        examples[(name, "Nhãn thật")] = (X, y)
                    rows.append({
                        "dataset": name, "seed": seed, "method": method,
                        "ari": float(adjusted_rand_score(y, labels)),
                        "nmi": float(normalized_mutual_info_score(y, labels)),
                        "n_clusters": len(set(labels.tolist()) - {-1}),
                        "noise_fraction": float(np.mean(labels == -1)),
                        "runtime_seconds": elapsed,
                    })
                print(f"[{name}] seed {seed + 1}/{args.seeds} xong", flush=True)

    pd.DataFrame(param_rows).to_csv(OUT / "e2_params.csv", index=False)
    pd.concat(grids).to_csv(OUT / "e2_selection_params.csv", index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "e2_runs.csv", index=False)

    eval_runs = runs[runs.seed >= 1] if args.seeds > 1 else runs
    summary = eval_runs.groupby(["dataset", "method"], sort=False).agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
        n_clusters_mean=("n_clusters", "mean"), n_clusters_std=("n_clusters", "std"),
        noise_fraction_mean=("noise_fraction", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"), runtime_seconds_std=("runtime_seconds", "std"),
    ).reset_index()
    summary.to_csv(OUT / "e2_summary.csv", index=False)

    cols = ["Nhãn thật"] + METHODS
    fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(3.3 * len(cols), 3.3 * len(DATASETS)),
                             constrained_layout=True)
    for ri, name in enumerate(DATASETS):
        X, y = examples[(name, "Nhãn thật")]
        P = X
        for ci, method in enumerate(cols):
            labels = examples[(name, method)][1]
            ax = axes[ri, ci]
            ax.scatter(P[:, 0], P[:, 1], c=labels, s=7, cmap="tab20", linewidths=0)
            if method == "Nhãn thật":
                ax.set_title(f"{name} — nhãn thật\nk = {TRUE_K[name]}", fontsize=9)
            else:
                row = runs[(runs.dataset == name) & (runs.seed == 0) & (runs.method == method)].iloc[0]
                ax.set_title(f"{method}\nARI = {row.ari:.3f}, số cụm = {row.n_clusters}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "e2_grid.png", dpi=150)
    plt.close(fig)

    mean = summary.pivot(index="dataset", columns="method", values="ari_mean").reindex(index=DATASETS, columns=METHODS)
    std = summary.pivot(index="dataset", columns="method", values="ari_std").reindex(index=DATASETS, columns=METHODS)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    mean.plot(kind="bar", yerr=std.fillna(0), ax=ax, capsize=2, rot=0)
    seeds_used = f"mẫu con seed 1–{args.seeds - 1}" if args.seeds > 1 else "toàn bộ dữ liệu"
    ax.set_title(f"ARI trung bình theo bộ dữ liệu ({seeds_used})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ARI"); ax.set_xlabel("Bộ dữ liệu"); ax.legend(fontsize=8)
    fig.savefig(OUT / "e2_ari_bar.png", dpi=150)
    plt.close(fig)

    metadata = {
        "experiment": "E2 – so sánh thuật toán", "seeds": list(range(args.seeds)),
        "summary_seeds": sorted(eval_runs.seed.unique().tolist()), "datasets": DATASETS,
        "selection": "α (Mean-Shift) và eps (DBSCAN) chọn theo silhouette cao nhất trên seed 0, không dùng nhãn; hoà thì lấy giá trị nhỏ hơn; giữ nguyên cho các seed khác",
        "selection_constraints": f"chỉ xét cấu hình có ≥ 2 cụm và ≥ {MIN_CORE_FRACTION:.0%} điểm không phải nhiễu; silhouette tính trên các điểm không phải nhiễu",
        "alpha_grid": ALPHA_GRID, "eps_grid": EPS_GRID, "min_samples": MIN_SAMPLES,
        "selected_params": params, "true_k": TRUE_K,
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True), h0 = meanshift.estimate_bandwidth(X, quantile=0.3)",
        "oracle_k": "KMeans, GMM được biết trước số cụm đúng",
        "data": f"sklearn.datasets (Iris, Wine, Breast Cancer), chuẩn hoá rồi PCA 2D; seed 0 = toàn bộ dữ liệu (dùng chọn tham số), seed ≥ 1 = mẫu con phân tầng {SUBSAMPLE:.0%}",
        "standardization": "StandardScaler rồi PCA 2 thành phần, làm riêng cho từng (bộ, seed)",
        "plot": "Hình lưới vẽ trực tiếp trên không gian 2D mà các thuật toán sử dụng",
        "noise": "Nhãn -1 của DBSCAN được tính như một nhãn trong ARI/NMI; không đếm vào số cụm",
        "threadpool_limit": 1,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (OUT / "e2_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nARI trung bình (" + seeds_used + "):")
    print(mean.round(3).to_string())
    print(f"\nĐã lưu kết quả E2 tại {OUT}")


if __name__ == "__main__":
    main()
