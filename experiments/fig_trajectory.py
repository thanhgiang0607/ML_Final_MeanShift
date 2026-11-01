import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity

from common import ROOT, load_real

sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom

N_SHOW = 40


def main():
    X, _ = load_real("iris")
    h = 0.5 * estimate_bandwidth(X, quantile=0.3)
    model = MeanShiftCustom(bandwidth=h, record_trajectories=True).fit(X)
    trajectories = model.trajectories_
    start = int(np.argmax([len(t) for t in trajectories]))
    ends = np.array([t[-1] for t in trajectories])
    seed_cluster = np.argmin(((ends[:, None] - model.cluster_centers_[None]) ** 2).sum(-1), axis=1)

    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    density = np.exp(KernelDensity(kernel="epanechnikov", bandwidth=h).fit(X).score_samples(grid)).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    filled = ax.contourf(gx, gy, density, levels=20, cmap="Greys", alpha=0.6)
    fig.colorbar(filled, ax=ax, label="Mật độ KDE Epanechnikov")
    ax.scatter(X[:, 0], X[:, 1], s=3, c="0.35", alpha=0.35, label="Điểm dữ liệu")

    rng = np.random.default_rng(0)
    shown = rng.choice(np.setdiff1d(np.arange(len(X)), [start]), size=N_SHOW, replace=False)
    for i in shown:
        t = trajectories[i]
        ax.plot(t[:, 0], t[:, 1], ".-", color=f"C{seed_cluster[i]}", lw=1, ms=3, alpha=0.85)
        ax.plot(*t[0], "o", mfc="none", mec=f"C{seed_cluster[i]}", ms=5)
    t0 = trajectories[start]
    ax.plot(t0[:, 0], t0[:, 1], "o-", color="magenta", lw=2.4, ms=6, zorder=4,
            label=f"Quỹ đạo dài nhất, từ X[{start}] ({len(t0) - 1} bước)")
    ax.scatter(*model.cluster_centers_.T, marker="X", s=180, c="black", edgecolors="white",
               linewidths=1.5, zorder=5, label=f"Tâm cụm ({len(model.cluster_centers_)} mode)")
    ax.plot([], [], ".-", color="C0", label=f"{N_SHOW} quỹ đạo ngẫu nhiên (màu = cụm hội tụ)")

    ax.set_title(f"Quỹ đạo dịch chuyển của MeanShiftCustom\nIris (PCA 2D), nhân phẳng, h = {h:.2f}", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    fig.tight_layout()
    out = ROOT / "report" / "figures" / "trajectory_iris.png"
    fig.savefig(out, dpi=150)
    steps = np.array([len(t) - 1 for t in trajectories])
    print(f"Đã lưu {out}")
    print(f"h = {h:.3f}, {len(model.cluster_centers_)} cụm, số bước: trung vị {np.median(steps):.0f}, "
          f"lớn nhất {steps.max()}; X[{start}]: {len(t0) - 1} bước")


if __name__ == "__main__":
    main()
