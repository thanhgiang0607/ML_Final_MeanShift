import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATASETS, ROOT, TRUE_K, load_raw, reduce_2d

CLASS_NAMES = {
    "iris": ["setosa", "versicolor", "virginica"],
    "wine": ["giống 0", "giống 1", "giống 2"],
    "breast_cancer": ["ác tính", "lành tính"],
}


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.6))
    for ax, name in zip(axes, DATASETS):
        X_raw, y = load_raw(name)
        P, explained = reduce_2d(X_raw)
        for k, label in enumerate(CLASS_NAMES[name]):
            ax.scatter(P[y == k, 0], P[y == k, 1], s=10, alpha=0.75, color=f"C{k}", label=label)
        n, d = X_raw.shape
        ax.set_title(f"{name}: n = {n}, {d} chiều gốc → 2D, k = {TRUE_K[name]}\nPCA 2D giữ {explained:.0%} phương sai",
                     fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle("Ba bộ dữ liệu thật sau khi chuẩn hoá và giảm về 2 chiều bằng PCA (màu = lớp thật)")
    fig.tight_layout()
    out = ROOT / "report" / "figures" / "datasets.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
