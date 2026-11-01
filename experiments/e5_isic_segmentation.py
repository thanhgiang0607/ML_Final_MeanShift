import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isic_baselines import gmm_labels, kmeans_labels, otsu_labels
from isic_utils import (BORDER_SHARE, BORDER_WIDTH, MAX_SIDE, MIN_AREA, SPLITS,
                        dice, features, iou, labels_to_mask, load_image, load_split)
from meanshift import MeanShiftCustom

E4_BEST = ROOT / "results" / "e4" / "best_h_3d_lab.json"
OUT = ROOT / "results" / "e5"

METHODS = ["otsu", "kmeans", "gmm", "ms_rgb3", "ms_lab3", "ms_lab5"]
MS_METHODS = ["ms_rgb3", "ms_lab3", "ms_lab5"]
NAMES = {"otsu": "Otsu (L)", "kmeans": "K-Means k=2", "gmm": "GMM k=2",
         "ms_rgb3": "MS 3D RGB", "ms_lab3": "MS 3D Lab", "ms_lab5": "MS 5D Lab"}

RGB_HS = [12, 16, 20, 24, 32, 40, 48]
LAB5_HS = [8, 16, 32, 64]
LAB5_HR = [4, 8, 12, 16, 20]

QUICK_IMAGES = 10
TEST_REPEAT = 3
N_SKLEARN = 20
N_FAILURES = 8


def ms_features(rgb, method, params):
    if method == "ms_rgb3":
        return features(rgb, "rgb3"), float(params["h"])
    if method == "ms_lab3":
        return features(rgb, "lab3"), float(params["h"])
    if method == "ms_lab5":
        return features(rgb, "lab5", hs=params["hs"], hr=params["hr"]), 1.0
    raise ValueError(method)


def compute_labels(rgb, method, params):
    if method == "otsu":
        return otsu_labels(rgb)
    if method == "kmeans":
        return kmeans_labels(features(rgb, "lab3"))
    if method == "gmm":
        return gmm_labels(features(rgb, "lab3"))
    X, h = ms_features(rgb, method, params)
    return MeanShiftCustom(bandwidth=h, bin_seeding=True, neighbor_search="kdtree").fit(X).labels_


def run_one(image_id, rgb, gt, method, params, repeat=1):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        labels = np.asarray(compute_labels(rgb, method, params))
        times.append(time.perf_counter() - start)
    pred = labels_to_mask(labels, rgb)
    return {
        "image_id": image_id, "method": method, "params": params_label(method, params),
        "dice": dice(pred, gt), "iou": iou(pred, gt), "empty_mask": bool(not pred.any()),
        "n_clusters": int(len(np.unique(labels[labels != -1]))),
        "time_seconds": float(np.median(times)),
    }


def params_label(method, params):
    if method == "ms_lab5":
        return f"hs={params['hs']:g},hr={params['hr']:g}"
    if method in ("ms_rgb3", "ms_lab3"):
        return f"h={params['h']:g}"
    return "-"


def e4_best_h():
    if not E4_BEST.is_file():
        raise FileNotFoundError(f"Thiếu {E4_BEST}. Chạy trước: python3 experiments/e4_isic.py")
    return float(json.loads(E4_BEST.read_text(encoding="utf-8"))["best_h"])


def tune_grid():
    grid = [(m, {}) for m in ("otsu", "kmeans", "gmm")]
    grid += [("ms_rgb3", {"h": h}) for h in RGB_HS]
    grid += [("ms_lab3", {"h": e4_best_h()})]
    grid += [("ms_lab5", {"hs": hs, "hr": hr}) for hs in LAB5_HS for hr in LAB5_HR]
    return grid


def summarize(runs):
    summary = (runs.groupby(["method", "params"], as_index=False, sort=False)
               .agg(n_images=("image_id", "nunique"),
                    dice_mean=("dice", "mean"), dice_std=("dice", "std"), dice_median=("dice", "median"),
                    iou_mean=("iou", "mean"), iou_std=("iou", "std"),
                    fail_rate=("dice", lambda d: float((d < 0.1).mean())),
                    empty_mask_rate=("empty_mask", "mean"),
                    n_clusters_median=("n_clusters", "median"),
                    time_median=("time_seconds", "median"), time_mean=("time_seconds", "mean")))
    std_columns = [c for c in summary.columns if c.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)
    return summary


def stage_tune(image_ids, out):
    grid = tune_grid()
    rows = []
    for i, image_id in enumerate(image_ids, start=1):
        print(f"[tune {i}/{len(image_ids)}] {image_id}", flush=True)
        rgb, gt = load_image(image_id)
        rows += [run_one(image_id, rgb, gt, m, p) for m, p in grid]
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    runs.to_csv(out / "e5_tune_runs.csv", index=False)
    summary.to_csv(out / "e5_tune_summary.csv", index=False)

    by_label = {(m, params_label(m, p)): p for m, p in grid}
    best = {}
    for method in METHODS:
        rows_m = summary[summary.method == method]
        top = rows_m.loc[rows_m.dice_mean.idxmax()]
        best[method] = {"params": by_label[(method, top.params)], "label": top.params,
                        "tune_dice_mean": float(top.dice_mean), "tune_dice_std": float(top.dice_std)}
    ms_best = max(MS_METHODS, key=lambda m: best[m]["tune_dice_mean"])
    record = {"n_tune_images": len(image_ids), "methods": best, "best_meanshift": ms_best,
              "note": "Chọn theo Dice trung bình cao nhất trên tập tune. ms_lab3 lấy h từ E4, không chỉnh lại."}
    (out / "e5_best_params.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== E5 – TUNE =====")
    print(summary[["method", "params", "dice_mean", "dice_std", "fail_rate", "empty_mask_rate",
                   "n_clusters_median", "time_median"]].round(3).to_string(index=False))
    print("\nTham số tốt nhất:")
    for method in METHODS:
        print(f"  {NAMES[method]:12s} {best[method]['label']:14s} Dice {best[method]['tune_dice_mean']:.3f}")
    print(f"Mean-Shift tốt nhất trên tune: {NAMES[ms_best]}")


def load_best(out):
    path = out / "e5_best_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: --stage tune")
    return json.loads(path.read_text(encoding="utf-8"))


def wilcoxon_table(runs, ms_best):
    pivot = runs.pivot(index="image_id", columns="method", values="dice")
    rows = []
    for other in METHODS:
        if other == ms_best:
            continue
        diff = pivot[ms_best] - pivot[other]
        try:
            stat, p = wilcoxon(pivot[ms_best], pivot[other])
        except ValueError:
            stat, p = np.nan, 1.0
        rows.append({"meanshift": ms_best, "other": other,
                     "dice_meanshift": float(pivot[ms_best].mean()), "dice_other": float(pivot[other].mean()),
                     "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
                     "n_better": int((diff > 1e-9).sum()), "n_worse": int((diff < -1e-9).sum()),
                     "n_tie": int((diff.abs() <= 1e-9).sum()), "statistic": float(stat), "p_value": float(p)})
    table = pd.DataFrame(rows)
    order = np.argsort(table.p_value.to_numpy())
    m = len(table)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * table.p_value.iloc[idx]))
        adjusted[idx] = running
    table["p_holm"] = adjusted
    return table


def vs_sklearn(image_ids, best):
    rows = []
    for image_id in image_ids:
        rgb, gt = load_image(image_id)
        for method in MS_METHODS:
            X, h = ms_features(rgb, method, best["methods"][method]["params"])
            ours = MeanShiftCustom(bandwidth=h, bin_seeding=True, neighbor_search="kdtree").fit(X)
            ref = MeanShift(bandwidth=h, bin_seeding=True, n_jobs=1).fit(X)
            mask_ours, mask_ref = labels_to_mask(ours.labels_, rgb), labels_to_mask(ref.labels_, rgb)
            rows.append({"image_id": image_id, "method": method,
                         "n_clusters_custom": len(ours.cluster_centers_),
                         "n_clusters_sklearn": len(ref.cluster_centers_),
                         "ari_labels": adjusted_rand_score(ref.labels_, ours.labels_),
                         "dice_masks": dice(mask_ours, mask_ref),
                         "dice_gt_custom": dice(mask_ours, gt), "dice_gt_sklearn": dice(mask_ref, gt)})
    return pd.DataFrame(rows)


def plot_box(runs, path, n_images):
    data = [runs[runs.method == m].dice.to_numpy() for m in METHODS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(data, showmeans=True, meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black"})
    ax.set_xticks(range(1, len(METHODS) + 1), [NAMES[m] for m in METHODS])
    for i, d in enumerate(data, start=1):
        ax.text(i, 1.03, f"{d.mean():.3f}", ha="center", fontsize=9)
    ax.set_ylim(-0.03, 1.09)
    ax.set_ylabel("Dice")
    ax.set_title(f"Dice trên {n_images} ảnh test (▲ và số phía trên: trung bình)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mask_panels(image_ids, tags, runs, best, path, title):
    fig, axes = plt.subplots(len(image_ids), 2 + len(METHODS),
                             figsize=(2.3 * (2 + len(METHODS)), 2.2 * len(image_ids)), squeeze=False)
    for row, image_id, tag in zip(axes, image_ids, tags):
        rgb, gt = load_image(image_id)
        row[0].imshow(rgb)
        row[0].set_title(f"{image_id}\n{tag}", fontsize=8)
        row[1].imshow(gt, cmap="gray")
        row[1].set_title("Mask bác sĩ", fontsize=8)
        for ax, method in zip(row[2:], METHODS):
            pred = labels_to_mask(compute_labels(rgb, method, best["methods"][method]["params"]), rgb)
            ax.imshow(rgb)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1)
            if pred.any():
                ax.contour(pred, levels=[0.5], colors="red", linewidths=1)
            ax.set_title(f"{NAMES[method]}\nDice {dice(pred, gt):.2f}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title + "  (viền xanh: bác sĩ, viền đỏ: dự đoán)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def stage_test(image_ids, out):
    best = load_best(out)
    ms_best = best["best_meanshift"]
    rows = []
    for i, image_id in enumerate(image_ids, start=1):
        print(f"[test {i}/{len(image_ids)}] {image_id}", flush=True)
        rgb, gt = load_image(image_id)
        rows += [run_one(image_id, rgb, gt, m, best["methods"][m]["params"], repeat=TEST_REPEAT)
                 for m in METHODS]
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    summary["tune_dice_mean"] = summary.method.map(lambda m: best["methods"][m]["tune_dice_mean"])
    wil = wilcoxon_table(runs, ms_best)
    print("So với sklearn MeanShift ...", flush=True)
    sk = vs_sklearn(image_ids[:N_SKLEARN], best)

    runs.to_csv(out / "e5_test_runs.csv", index=False)
    summary.to_csv(out / "e5_test_summary.csv", index=False)
    wil.to_csv(out / "e5_wilcoxon.csv", index=False)
    sk.to_csv(out / "e5_vs_sklearn.csv", index=False)

    plot_box(runs, out / "e5_dice_box.png", len(image_ids))
    ranked = runs[runs.method == ms_best].sort_values(["dice", "image_id"])
    picks = [ranked.iloc[-1], ranked.iloc[len(ranked) // 2], ranked.iloc[0]]
    mask_panels([r.image_id for r in picks],
                [f"{t} ({NAMES[ms_best]} Dice {r.dice:.2f})" for t, r in zip(("tốt nhất", "trung vị", "tệ nhất"), picks)],
                runs, best, out / "e5_examples.png", f"Ví dụ phân đoạn trên tập test, xếp theo {NAMES[ms_best]}")
    worst = ranked.head(N_FAILURES)
    mask_panels(list(worst.image_id), [f"{NAMES[ms_best]} Dice {d:.2f}" for d in worst.dice],
                runs, best, out / "e5_failures.png", f"{N_FAILURES} ảnh test tệ nhất của {NAMES[ms_best]}")

    print("\n===== E5 – TEST =====")
    print(summary[["method", "params", "dice_mean", "dice_std", "dice_median", "iou_mean", "fail_rate",
                   "time_median", "tune_dice_mean"]].round(3).to_string(index=False))
    print(f"\nWilcoxon ({NAMES[ms_best]} so với từng phương pháp):")
    print(wil[["other", "mean_diff", "n_better", "n_worse", "p_value", "p_holm"]].round(4).to_string(index=False))
    print("\nSo với sklearn (trung bình trên", sk.image_id.nunique(), "ảnh):")
    print(sk.groupby("method")[["ari_labels", "dice_masks"]].agg(["mean", "min"]).round(4).to_string())


def write_metadata(out, stage, image_ids):
    path = out / "e5_metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    split = SPLITS / f"isic_{stage}.txt"
    meta.update({
        "experiment": "E5 – phân đoạn tổn thương da ISIC 2018",
        "methods": NAMES,
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "baselines": "isic_baselines: Otsu kênh L; K-Means, GMM k=2 trên 3D Lab, random_state=0",
        "features": f"isic_utils, ảnh thu nhỏ cạnh dài {MAX_SIDE} px",
        "mask_rule": f"isic_utils.labels_to_mask (border_width={BORDER_WIDTH}, border_share={BORDER_SHARE}, min_area={MIN_AREA})",
        "tune_grid": {"ms_rgb3_h": RGB_HS, "ms_lab3_h": "từ results/e4/best_h_3d_lab.json",
                      "ms_lab5_hs": LAB5_HS, "ms_lab5_hr": LAB5_HR},
        "timing": f"time.perf_counter từ ảnh RGB tới nhãn (gồm tạo đặc trưng); tune 1 lần, test trung vị {TEST_REPEAT} lần",
        f"{stage}_n_images": len(image_ids),
        f"{stage}_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "machine": f"{platform.system()} {platform.machine()}",
    })
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E5: phân đoạn tổn thương da ISIC bằng Mean-Shift và baseline")
    parser.add_argument("--stage", choices=["tune", "test"], required=True)
    parser.add_argument("--quick", action="store_true", help=f"Chỉ {QUICK_IMAGES} ảnh, ghi ra thư mục tạm")
    args = parser.parse_args()

    image_ids = load_split(args.stage)
    out = OUT
    if args.quick:
        image_ids = image_ids[:QUICK_IMAGES]
        out = Path(tempfile.gettempdir()) / "e5_quick"
        print(f"[--quick] {len(image_ids)} ảnh, ghi vào {out}")
        if args.stage == "test" and not (out / "e5_best_params.json").is_file() and (OUT / "e5_best_params.json").is_file():
            out.mkdir(parents=True, exist_ok=True)
            (out / "e5_best_params.json").write_text((OUT / "e5_best_params.json").read_text(encoding="utf-8"),
                                                     encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "tune":
        stage_tune(image_ids, out)
    else:
        stage_test(image_ids, out)
    write_metadata(out, args.stage, image_ids)


if __name__ == "__main__":
    main()
