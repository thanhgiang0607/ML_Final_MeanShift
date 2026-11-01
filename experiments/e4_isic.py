import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import ROOT, timeit
from meanshift.bandwidth import estimate_bandwidth
from meanshift.core import MeanShiftCustom

QUANTILE = 0.3

ISIC_HS = [2, 4, 6, 8, 10, 12, 16, 20]
AUTO = "auto"
ISIC_N_SAMPLES = 500
ISIC_REPEAT = 3
QUICK_IMAGES = 10
N_EXAMPLE_LEVELS = 4


def isic_settings(h_auto):
    return [(str(h), float(h)) for h in ISIC_HS] + [(AUTO, float(h_auto))]


def fit_labels(X, h):
    return MeanShiftCustom(bandwidth=h, bin_seeding=True, neighbor_search="kdtree").fit(X)


def run_isic_image(image_id, repeat=ISIC_REPEAT):
    from isic_utils import dice, features, iou, labels_to_mask, load_image

    rgb, gt = load_image(image_id)
    X = features(rgb, "lab3")

    h_auto = estimate_bandwidth(
        X, quantile=QUANTILE, n_samples=ISIC_N_SAMPLES, random_state=0
    )
    if not np.isfinite(h_auto) or h_auto <= 0:
        raise ValueError(f"Bandwidth tự động không hợp lệ: ảnh={image_id}, h={h_auto}")

    rows = []
    for setting, h in isic_settings(h_auto):
        model, seconds = timeit(lambda h=h: fit_labels(X, h), repeat=repeat)
        labels = np.asarray(model.labels_)
        pred = labels_to_mask(labels, rgb)
        rows.append({
            "image_id": image_id,
            "h_setting": setting,
            "h": h,
            "n_clusters": int(len(np.unique(labels[labels != -1]))),
            "dice": dice(pred, gt),
            "iou": iou(pred, gt),
            "empty_mask": bool(not pred.any()),
            "time_seconds": seconds,
        })
    return rows


def summarize_isic(runs):
    summary = (
        runs.groupby("h_setting", as_index=False)
        .agg(
            n_images=("image_id", "nunique"),
            h_mean=("h", "mean"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_median=("dice", "median"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_median=("n_clusters", "median"),
            empty_mask_rate=("empty_mask", "mean"),
            time_mean=("time_seconds", "mean"),
            time_std=("time_seconds", "std"),
        )
    )
    std_columns = [col for col in summary.columns if col.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)

    summary["_order"] = summary["h_setting"].map(lambda s: np.inf if s == AUTO else float(s))
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def best_fixed_h(summary):
    fixed = summary[summary["h_setting"] != AUTO]
    return fixed.loc[fixed["dice_mean"].idxmax()]


def plot_isic_h(summary, output_path, n_images):
    fixed = summary[summary["h_setting"] != AUTO]
    auto = summary[summary["h_setting"] == AUTO].iloc[0]
    best = best_fixed_h(summary)
    x = fixed["h_mean"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(x, fixed["dice_mean"], marker="o", color="C0", label="Dice (trung bình)")
    ax.fill_between(
        x,
        np.clip(fixed["dice_mean"] - fixed["dice_std"], 0, 1),
        np.clip(fixed["dice_mean"] + fixed["dice_std"], 0, 1),
        color="C0", alpha=0.15, label="Dice ± 1 độ lệch chuẩn",
    )
    ax.plot(x, fixed["iou_mean"], marker="s", color="C1", label="IoU (trung bình)")
    ax.axhline(auto["dice_mean"], color="C0", linestyle="--", linewidth=1.2,
               label=f"Dice, h = estimate_bandwidth ({auto['dice_mean']:.3f})")
    ax.axvline(best["h_mean"], color="gray", linestyle=":", linewidth=1.2)
    ax.annotate(f"h tốt nhất = {best['h_mean']:g}\nDice = {best['dice_mean']:.3f}",
                xy=(best["h_mean"], best["dice_mean"]),
                xytext=(25, 40), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "gray"})
    ax.set_title(f"Độ nhạy Dice/IoU theo h ({n_images} ảnh tune, 3D Lab)")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Dice / IoU")
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(x, fixed["n_clusters_mean"], marker="o", color="C2", label="Trung bình")
    ax.plot(x, fixed["n_clusters_median"], marker="^", color="C3", linestyle="-", label="Trung vị")
    ax.axhline(auto["n_clusters_median"], color="C3", linestyle="--", linewidth=1.2,
               label=f"Trung vị, h = estimate_bandwidth ({auto['n_clusters_median']:g})")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 100, 200])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("Số cụm theo h")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Số cụm mỗi ảnh (thang log)")

    ax = axes[2]
    ax.plot(x, fixed["time_mean"], marker="o", color="C4", label="Thời gian trung bình / ảnh")
    ax.fill_between(
        x,
        np.maximum(fixed["time_mean"] - fixed["time_std"], 0),
        fixed["time_mean"] + fixed["time_std"],
        color="C4", alpha=0.15, label="± 1 độ lệch chuẩn",
    )
    ax.axhline(auto["time_mean"], color="C4", linestyle="--", linewidth=1.2,
               label=f"h = estimate_bandwidth ({auto['time_mean']:.3f} s)")
    ax.set_title("Thời gian chạy theo h")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Thời gian (giây)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_levels(best_h):
    idx = np.linspace(0, len(ISIC_HS) - 1, N_EXAMPLE_LEVELS).round().astype(int)
    levels = [ISIC_HS[i] for i in idx]
    levels[int(np.argmin([abs(v - best_h) for v in levels]))] = best_h
    return sorted(set(levels))


def plot_isic_examples(runs, best_h, output_path):
    from isic_utils import dice, features, labels_to_mask, load_image

    at_best = runs[runs["h_setting"] == f"{best_h:g}"].sort_values(["dice", "image_id"])
    picks = [("trung vị", at_best.iloc[len(at_best) // 2]["image_id"])]
    if len(at_best) > 1:
        picks.append(("tệ nhất", at_best.iloc[0]["image_id"]))

    levels = example_levels(best_h)
    fig, axes = plt.subplots(2 * len(picks), 1 + len(levels),
                             figsize=(3 * (1 + len(levels)), 6 * len(picks)), squeeze=False)

    for p, (tag, image_id) in enumerate(picks):
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        top, bottom = axes[2 * p], axes[2 * p + 1]
        top[0].imshow(rgb)
        top[0].set_title(f"{image_id}\n(ảnh {tag}, Dice tại h = {best_h:g})", fontsize=9)
        bottom[0].imshow(gt, cmap="gray")
        bottom[0].set_title("Mask bác sĩ", fontsize=9)
        for j, h in enumerate(levels, start=1):
            labels = np.asarray(fit_labels(X, float(h)).labels_)
            pred = labels_to_mask(labels, rgb)
            n_clusters = len(np.unique(labels[labels != -1]))
            top[j].imshow(labels.reshape(rgb.shape[:2]), cmap="tab20", interpolation="nearest")
            top[j].set_title(f"h = {h:g}: {n_clusters} cụm", fontsize=9)
            bottom[j].imshow(pred, cmap="gray")
            bottom[j].set_title(f"Mask dự đoán · Dice {dice(pred, gt):.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Mean-Shift 3D Lab: ảnh hưởng của bandwidth h đến cụm và mask", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_isic(quick=False):
    from isic_utils import (BORDER_SHARE, BORDER_WIDTH, MAX_SIDE, MIN_AREA,
                            SPLITS, load_split)

    image_ids = load_split("tune")
    if quick:
        image_ids = image_ids[:QUICK_IMAGES]
        output_dir = Path(tempfile.gettempdir()) / "e4_isic_quick"
        print(f"[--quick] chỉ chạy {len(image_ids)} ảnh; kết quả ghi vào {output_dir}, không đụng results/e4/")
    else:
        output_dir = ROOT / "results" / "e4"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, image_id in enumerate(image_ids, start=1):
        print(f"[{i}/{len(image_ids)}] ảnh={image_id}")
        all_rows.extend(run_isic_image(image_id))

    runs = pd.DataFrame(all_rows)
    summary = summarize_isic(runs)
    best = best_fixed_h(summary)
    auto = summary[summary["h_setting"] == AUTO].iloc[0]
    best_h = float(best["h_setting"])

    runs_path = output_dir / "e4_isic_runs.csv"
    summary_path = output_dir / "e4_isic_summary.csv"
    plot_path = output_dir / "e4_isic_h.png"
    examples_path = output_dir / "e4_isic_examples.png"
    best_path = output_dir / "best_h_3d_lab.json"
    metadata_path = output_dir / "e4_isic_metadata.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_isic_h(summary, plot_path, len(image_ids))
    plot_isic_examples(runs, best_h, examples_path)

    best_record = {
        "feature": "lab3",
        "unit": "Lab",
        "best_h": best_h,
        "dice_mean": float(best["dice_mean"]),
        "dice_std": float(best["dice_std"]),
        "iou_mean": float(best["iou_mean"]),
        "n_clusters_median": float(best["n_clusters_median"]),
        "n_images": int(best["n_images"]),
        "grid": {r["h_setting"]: round(float(r["dice_mean"]), 4) for _, r in summary.iterrows()},
        "auto_bandwidth": {
            "dice_mean": float(auto["dice_mean"]),
            "mean_h": float(auto["h_mean"]),
            "quantile": QUANTILE,
            "n_samples": ISIC_N_SAMPLES,
        },
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "note": ("h chọn theo Dice trung bình cao nhất trên tập tune; "
                 "kết quả cuối cùng phải báo cáo trên tập test (E5)."
                 + (" CHẠY --quick: chỉ một phần ảnh tune, không dùng làm kết quả chính." if quick else "")),
    }
    best_path.write_text(json.dumps(best_record, indent=2, ensure_ascii=False), encoding="utf-8")

    split_file = SPLITS / "isic_tune.txt"
    metadata = {
        "experiment": "E4 – độ nhạy theo bandwidth (phần 2: ISIC tune)",
        "quick": bool(quick),
        "n_images": len(image_ids),
        "h_grid_lab": ISIC_HS,
        "h_auto": f"estimate_bandwidth(X, quantile={QUANTILE}, n_samples={ISIC_N_SAMPLES}, random_state=0) trên từng ảnh",
        "features": "3D Lab (isic_utils.features(rgb, 'lab3')), ảnh thu nhỏ cạnh dài về "
                    f"{MAX_SIDE} px",
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "mask_rule": (f"isic_utils.labels_to_mask (border_width={BORDER_WIDTH}, "
                      f"border_share={BORDER_SHARE}, min_area={MIN_AREA})"),
        "timing": f"time.perf_counter quanh fit, {ISIC_REPEAT} lần, lấy trung vị",
        "best_h": "h cố định có Dice trung bình cao nhất trên các ảnh đã chạy",
        "input_sha256": {
            f"data/splits/{split_file.name}": hashlib.sha256(split_file.read_bytes()).hexdigest()
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== TÓM TẮT E4 – ISIC TUNE =====")
    print(f"Số ảnh: {len(image_ids)} · số lần chạy: {len(runs)}")
    print(f"Kết quả: {output_dir}")
    print(summary[[
        "h_setting", "h_mean", "dice_mean", "dice_std", "iou_mean",
        "n_clusters_median", "empty_mask_rate", "time_mean",
    ]].round(3).to_string(index=False))
    print(f"\nh tốt nhất (Lab): {best_h:g} · Dice trung bình {best['dice_mean']:.3f} "
          f"· h = estimate_bandwidth: Dice {auto['dice_mean']:.3f}")
    print(f"=> A dùng cho E5: {best_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="E4 phần 2: độ nhạy của Mean-Shift 3D Lab theo bandwidth trên ISIC tune."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Chỉ chạy {QUICK_IMAGES} ảnh đầu, ghi kết quả ra thư mục tạm.",
    )
    run_isic(quick=parser.parse_args().quick)


if __name__ == "__main__":
    main()
