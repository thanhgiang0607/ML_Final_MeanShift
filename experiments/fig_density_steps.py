import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

PANELS = [
    ("flat", "epanechnikov_kde", "Nhân phẳng (flat)", "Mật độ KDE Epanechnikov"),
    ("gaussian", "gaussian_kde_direct", "Nhân Gauss", "Mật độ KDE Gauss"),
]


def main():
    frame = pd.read_csv(ROOT / "results" / "e1" / "e1_density_steps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (kernel, column, title, ylabel) in zip(axes, PANELS):
        subset = frame[frame.kernel == kernel]
        steps, density = subset.step.to_numpy(), subset[column].to_numpy()
        ax.plot(steps, density, "o-", color="C0" if kernel == "flat" else "C3")
        gain = density[-1] / density[0]
        ax.annotate(f"{density[0]:.4f} → {density[-1]:.4f} (×{gain:.2f})\nhội tụ sau {steps[-1]} bước",
                    xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_title(title)
        ax.set_xlabel("Bước dịch chuyển t")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Mật độ tại điểm đang dịch chuyển qua từng bước (Iris, PCA 2D, xuất phát từ X[{int(frame.start_index.iloc[0])}])")
    fig.tight_layout()
    out = ROOT / "report" / "figures" / "density_steps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Đã lưu {out}")
    for kernel, column, *_ in PANELS:
        values = frame[frame.kernel == kernel][column].to_numpy()
        print(f"{kernel}: {len(values) - 1} bước, mật độ tăng đơn điệu = {bool((values[1:] >= values[:-1]).all())}")


if __name__ == "__main__":
    main()
