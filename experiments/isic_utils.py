import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
ISIC = ROOT / "data" / "isic2018"
SPLITS = ROOT / "data" / "splits"

IMAGE_DIRS = [ISIC / "ISIC2018_Task1-2_Validation_Input", ISIC / "ISIC2018_Task1-2_Test_Input"]
MASK_DIRS = [ISIC / "ISIC2018_Task1_Validation_GroundTruth", ISIC / "ISIC2018_Task1_Test_GroundTruth"]

MAX_SIDE = 128
MODES = ("rgb3", "lab3", "lab5")
BORDER_WIDTH = 0.05
BORDER_SHARE = 0.5
MIN_AREA = 0.01


def load_split(name):
    if name not in ("tune", "test"):
        raise ValueError(f"name phải là 'tune' hoặc 'test', nhận {name!r}")
    path = SPLITS / f"isic_{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python3 data/download_isic.py")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find(image_id, folders, suffix):
    for folder in folders:
        path = folder / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Không tìm thấy {image_id}{suffix}. Chạy trước: python3 data/download_isic.py")


def load_image(image_id, max_side=MAX_SIDE):
    image = Image.open(_find(image_id, IMAGE_DIRS, ".jpg")).convert("RGB")
    mask = Image.open(_find(image_id, MASK_DIRS, "_segmentation.png")).convert("L")
    if image.size != mask.size:
        raise ValueError(f"{image_id}: ảnh {image.size} và mask {mask.size} khác kích thước")

    scale = max_side / max(image.size)
    if scale < 1:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask) > 127


def to_lab(rgb):
    return rgb2lab(np.asarray(rgb, dtype=np.uint8))


def features(rgb, mode, hs=None, hr=None):
    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận {mode!r}")
    h, w = rgb.shape[:2]
    if mode == "rgb3":
        return rgb.reshape(-1, 3).astype(np.float64)
    lab = to_lab(rgb).reshape(-1, 3)
    if mode == "lab3":
        return lab
    if hs is None or hr is None or hs <= 0 or hr <= 0:
        raise ValueError("mode 'lab5' cần hs > 0 và hr > 0")
    yy, xx = np.mgrid[0:h, 0:w]
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return np.hstack([xy / hs, lab / hr])


def border_ring(shape, width=BORDER_WIDTH):
    h, w = shape
    k = max(1, int(round(width * min(h, w))))
    ring = np.zeros((h, w), dtype=bool)
    ring[:k, :] = ring[-k:, :] = True
    ring[:, :k] = ring[:, -k:] = True
    return ring


def labels_to_mask(labels, rgb, border_width=BORDER_WIDTH, border_share=BORDER_SHARE,
                   min_area=MIN_AREA):
    h, w = rgb.shape[:2]
    labels = np.asarray(labels).reshape(h, w)
    lightness = to_lab(rgb)[..., 0]
    ring = border_ring((h, w), border_width)
    n_ring = ring.sum()

    best, best_l = None, np.inf
    for cluster in np.unique(labels):
        if cluster == -1:
            continue
        region = labels == cluster
        size = region.sum()
        on_ring = (region & ring).sum()
        if on_ring >= border_share * n_ring or on_ring >= border_share * size or size < min_area * h * w:
            continue
        mean_l = lightness[region].mean()
        if mean_l < best_l:
            best, best_l = cluster, mean_l
    if best is None:
        return np.zeros((h, w), dtype=bool)

    mask = ndimage.binary_fill_holes(labels == best)
    components, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = components == sizes.argmax()
    return mask


def dice(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def iou(pred, gt):
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def _check(n_images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    sys.path.insert(0, str(ROOT))
    from meanshift import MeanShiftCustom

    ids = load_split("tune")[:n_images]
    fig, axes = plt.subplots(len(ids), 4, figsize=(12, 3 * len(ids)), squeeze=False)
    for row, image_id in zip(axes, ids):
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        km = labels_to_mask(KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X), rgb)
        ms_model = MeanShiftCustom(bandwidth=8, bin_seeding=True).fit(X)
        ms = labels_to_mask(ms_model.labels_, rgb)
        n_ms = len(ms_model.cluster_centers_)
        print(f"{image_id} {rgb.shape[1]}x{rgb.shape[0]}  K-Means Dice={dice(km, gt):.3f}  "
              f"Mean-Shift h=8 ({n_ms} cụm) Dice={dice(ms, gt):.3f}")
        panels = [(rgb, image_id), (gt, "Mask thật"),
                  (km, f"K-Means k=2 · Dice {dice(km, gt):.2f}"),
                  (ms, f"Mean-Shift h=8 ({n_ms} cụm) · Dice {dice(ms, gt):.2f}")]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Kiểm tra isic_utils: đặc trưng 3D Lab, quy tắc labels_to_mask chung")
    fig.tight_layout()
    out = ROOT / "results" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "check_utils.png", dpi=150)
    plt.close(fig)
    print(f"Đã lưu {out / 'check_utils.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh isic_utils trên vài ảnh tune")
    parser.add_argument("--n", type=int, default=4, help="Số ảnh tune để kiểm tra (mặc định 4)")
    _check(parser.parse_args().n)
