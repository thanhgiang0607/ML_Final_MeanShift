import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isic_utils import (BORDER_SHARE, BORDER_WIDTH, MIN_AREA, border_ring, dice,
                        features, labels_to_mask, load_image, to_lab)
from meanshift import MeanShiftCustom

IMAGE_ID = "ISIC_0036281"
W, H = 17.0, 6.8

INK = "#222222"
MUTED = "#666666"
PANEL = "#F7F7F7"
EDGE = "#BBBBBB"
MS_COLOR = "#DCE9F7"
BASE_COLOR = "#EDEDED"
TUNE_COLOR = "#FFF1CC"
TEST_COLOR = "#DFF0D8"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text="", fc=PANEL, ec=EDGE, fs=9, weight="normal", ls="-", lw=1.0, r=0.08, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
                color=color, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def thumb(fig, x, y, w, h, img, cmap=None, title=None, contours=()):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.imshow(img, cmap=cmap, interpolation="nearest")
    for mask, color in contours:
        if mask.any():
            a.contour(mask, levels=[0.5], colors=color, linewidths=1.2)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(EDGE)
    if title:
        a.set_title(title, fontsize=8, color=MUTED, pad=3)
    return a


def mask_steps(labels, rgb):
    h, w = rgb.shape[:2]
    labels = labels.reshape(h, w)
    lightness = to_lab(rgb)[..., 0]
    ring = border_ring((h, w), BORDER_WIDTH)
    candidates, best, best_l = [], None, np.inf
    for c in np.unique(labels):
        region = labels == c
        size, on_ring = region.sum(), (region & ring).sum()
        if on_ring >= BORDER_SHARE * ring.sum() or on_ring >= BORDER_SHARE * size or size < MIN_AREA * h * w:
            continue
        candidates.append(c)
        if lightness[region].mean() < best_l:
            best, best_l = c, lightness[region].mean()
    return labels, ring, candidates, labels == best


def main():
    best = json.loads((ROOT / "results" / "e5" / "e5_best_params.json").read_text(encoding="utf-8"))
    p5 = best["methods"]["ms_lab5"]["params"]
    rgb, gt = load_image(IMAGE_ID)
    model = MeanShiftCustom(bandwidth=1.0, bin_seeding=True, neighbor_search="kdtree").fit(
        features(rgb, "lab5", hs=p5["hs"], hr=p5["hr"]))
    labels, ring, candidates, chosen = mask_steps(model.labels_, rgb)
    final = labels_to_mask(model.labels_, rgb)
    d = dice(final, gt)
    k = len(model.cluster_centers_)

    palette = plt.get_cmap("Set2").colors
    cmap = ListedColormap([palette[i % len(palette)] for i in range(k)])
    cand_img = np.ones(labels.shape + (3,)) * 0.85
    for c in candidates:
        cand_img[labels == c] = palette[c % len(palette)][:3]

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    top, ph = 2.05, 4.45
    cols = [(0.25, 2.3), (2.85, 2.75), (5.9, 3.55), (9.75, 4.6), (14.65, 2.1)]
    heads = ["(a) Đầu vào", "(b) Đặc trưng pixel", "(c) Phân cụm → nhãn từng pixel",
             "(d) Chọn cụm tổn thương (labels_to_mask)", "(e) Đánh giá"]
    for (x, w), head in zip(cols, heads):
        box(ax, x, top, w, ph, fc=PANEL)
        ax.text(x + 0.12, top + ph - 0.22, head, fontsize=10.5, weight="bold", color=INK, va="center")
    for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
        arrow(ax, x0 + w0 + 0.03, top + ph / 2, x1 - 0.03, top + ph / 2)

    tw, th = 1.9, 1.9 * rgb.shape[0] / rgb.shape[1]

    x, w = cols[0]
    thumb(fig, x + (w - tw) / 2, top + 2.1, tw, th, rgb)
    ax.text(x + w / 2, top + 1.35, "ISIC 2018 Task 1\nẢnh soi da + mask bác sĩ\nthu nhỏ cạnh dài 128 px\n(LANCZOS / NEAREST)",
            ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.4)

    x, w = cols[1]
    feats = [("RGB3", "(R, G, B)"), ("Lab3", "(L*, a*, b*)"),
             ("Lab5", "(x/hs, y/hs,\nL*/hr, a*/hr, b*/hr)"), ("L*", "(1 kênh, cho Otsu)")]
    fy = [top + 3.3, top + 2.45, top + 1.3, top + 0.35]
    fh = [0.62, 0.62, 0.9, 0.62]
    feat_mid = {}
    for (name, desc), y, hh in zip(feats, fy, fh):
        box(ax, x + 0.18, y, w - 0.36, hh, fc="white")
        ax.text(x + 0.32, y + hh / 2, name, fontsize=9.5, weight="bold", va="center", color=INK)
        ax.text(x + 0.95, y + hh / 2, desc, fontsize=8.3, va="center", color=MUTED, linespacing=1.3)
        feat_mid[name] = y + hh / 2

    x, w = cols[2]
    gx, gw = x + 0.15, 1.95
    box(ax, gx, top + 2.05, gw, 1.95, fc=MS_COLOR, ec="#9DB9DC")
    ax.text(gx + gw / 2, top + 3.78, "Mean-Shift (tự cài đặt)", fontsize=8.8, weight="bold", ha="center", color=INK)
    ax.text(gx + gw / 2, top + 3.53, "nhân phẳng · KD-Tree · bin seeding", fontsize=7.4, ha="center", color=MUTED)
    ms_items = ["MS-RGB3   h", "MS-Lab3   h = 8 (E4)", "MS-Lab5   hs, hr"]
    for i, t in enumerate(ms_items):
        ax.text(gx + 0.15, top + 3.15 - i * 0.36, t, fontsize=8.5, va="center", color=INK, family="DejaVu Sans Mono")
    box(ax, gx, top + 0.3, gw, 1.55, fc=BASE_COLOR, ec="#BDBDBD")
    ax.text(gx + gw / 2, top + 1.62, "Baseline", fontsize=8.8, weight="bold", ha="center", color=INK)
    for i, t in enumerate(["Otsu        L*", "K-Means     k = 2", "GMM         k = 2"]):
        ax.text(gx + 0.15, top + 1.27 - i * 0.33, t, fontsize=8.5, va="center", color=INK, family="DejaVu Sans Mono")
    thumb(fig, x + w - 1.42, top + 1.75, 1.3, 1.3 * th / tw, labels, cmap=cmap,
          title=f"MS-Lab5: {k} cụm")
    ax.text(x + w - 0.77, top + 1.25, "đầu ra: nhãn cụm\ncho từng pixel", fontsize=7.6, ha="center",
            va="center", color=MUTED)
    for name in ("RGB3", "Lab3", "Lab5"):
        arrow(ax, cols[1][0] + cols[1][1] - 0.18, feat_mid[name], gx, top + 2.9, color="#8FA9C9", lw=0.8)
    for name in ("Lab3", "L*"):
        arrow(ax, cols[1][0] + cols[1][1] - 0.18, feat_mid[name], gx, top + 1.0, color="#AAAAAA", lw=0.8)

    x, w = cols[3]
    sw = 1.3
    sh = sw * th / tw
    steps = [
        (cand_img, None, "① Loại cụm ở viền / góc tối\nvà cụm < 1% ảnh (xám)"),
        (np.where(chosen[..., None], rgb / 255, 0.92), None, "② Chọn cụm còn lại\ntối nhất theo L*"),
        (rgb, [(final, "red")], "③ Lấp lỗ, giữ vùng\nliên thông lớn nhất"),
    ]
    for i, (img, cont, cap) in enumerate(steps):
        sx = x + 0.2 + i * (sw + 0.2)
        thumb(fig, sx, top + 1.85, sw, sh, img, contours=cont or ())
        ax.text(sx + sw / 2, top + 1.4, cap, fontsize=7.8, ha="center", va="center", color=INK, linespacing=1.3)
        if i < 2:
            arrow(ax, sx + sw + 0.02, top + 1.85 + sh / 2, sx + sw + 0.18, top + 1.85 + sh / 2, color=MUTED, lw=1)
    ax.text(x + w / 2, top + 0.55, "Cùng một quy tắc cho cả 6 phương pháp\n→ chênh lệch Dice chỉ do thuật toán",
            fontsize=8.2, ha="center", va="center", color=MUTED, style="italic", linespacing=1.4)

    x, w = cols[4]
    ew = 1.7
    thumb(fig, x + (w - ew) / 2, top + 2.2, ew, ew * th / tw, rgb, contours=[(gt, "lime"), (final, "red")])
    ax.text(x + w / 2, top + 1.9, f"Dice = {d:.2f}", fontsize=9, ha="center", weight="bold", color=INK)
    ax.text(x + w / 2, top + 1.0, "xanh: bác sĩ · đỏ: dự đoán\n\nDice · IoU\nTỷ lệ thất bại (Dice < 0,1)\nThời gian / ảnh",
            fontsize=7.9, ha="center", va="center", color=MUTED, linespacing=1.35)

    by, bh = 0.3, 1.2
    ax.text(0.3, by + bh + 0.22, "Giao thức đánh giá", fontsize=10.5, weight="bold", color=INK, va="center")
    specs = [
        (2.6, TUNE_COLOR, "Tập tune: 100 ảnh\n(ISIC Validation)"),
        (4.0, TUNE_COLOR, "Dò lưới tham số, chạy (a)–(e)\nMS-RGB3: h ∈ {12, …, 48}\nMS-Lab5: hs ∈ {8, 16, 32, 64} × hr ∈ {4, …, 20}"),
        (3.0, "white", "Chốt tham số có Dice TB\ncao nhất (h = 24;\nhs = 32, hr = 20)"),
        (3.0, TEST_COLOR, "Tập test: 200 ảnh\n(ISIC Test), chạy 1 lần\nvới tham số đã chốt"),
        (4.55, TEST_COLOR, "Wilcoxon signed-rank theo cặp ảnh\n+ hiệu chỉnh Holm (5 phép so sánh)\nĐối chiếu scikit-learn · Phân tích lỗi"),
    ]
    stages, sx = [], 0.25
    for sw2, fc, text in specs:
        stages.append((sx, sw2, fc, text))
        sx += sw2 - 0.28 + 0.06
    for i, (sx, sw2, fc, text) in enumerate(stages):
        tip = 0.28
        pts = [(sx, by), (sx + sw2 - tip, by), (sx + sw2, by + bh / 2), (sx + sw2 - tip, by + bh),
               (sx, by + bh)]
        if i > 0:
            pts = pts + [(sx + tip, by + bh / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=EDGE, lw=1))
        ax.text(sx + sw2 / 2 + (0.1 if i > 0 else -0.05), by + bh / 2, text, ha="center", va="center",
                fontsize=8.4, color=INK, linespacing=1.4)
    ax.text(W - 0.25, by - 0.17, "Ảnh minh hoạ: " + IMAGE_ID + " (tập tune), MS-Lab5 với tham số đã chốt",
            fontsize=7.2, ha="right", color=MUTED)

    out = ROOT / "report" / "figures" / "pipeline_e5.png"
    fig.savefig(out, dpi=200)
    print(f"Đã lưu {out}  ({IMAGE_ID}: {k} cụm, Dice {d:.3f})")


if __name__ == "__main__":
    main()
