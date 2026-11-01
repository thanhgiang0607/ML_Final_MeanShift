import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATASETS, ROOT, TRUE_K, load_synthetic

DESCRIPTIONS = {
    "blobs": "4 cụm Gauss, std = 0.6",
    "blobs_varied": "3 cụm, std = 0.5 / 1.5 / 2.5",
    "blobs_aniso": "3 cụm bị kéo dẹt",
    "moons": "2 nửa vòng trăng, nhiễu 0.05",
    "circles": "2 vòng tròn đồng tâm, nhiễu 0.05",
}


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4.2))
    for ax, name in zip(axes, DATASETS):
        X, y = load_synthetic(name, 0)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", vmin=0, vmax=9, s=4, alpha=0.7)
        ax.set_title(f"{name} (k = {TRUE_K[name]})\n{DESCRIPTIONS[name]}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Năm bộ dữ liệu tổng hợp (seed 0, n = {len(X)} điểm/bộ, màu = nhãn đúng)")
    fig.tight_layout()
    out = ROOT / "report" / "figures" / "datasets.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
