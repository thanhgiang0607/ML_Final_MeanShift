import argparse

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from isic_utils import to_lab


def otsu_labels(rgb):
    lightness = to_lab(rgb)[..., 0].ravel()
    if lightness.max() == lightness.min():
        return np.zeros(lightness.size, dtype=int)
    return (lightness > threshold_otsu(lightness)).astype(int)


def kmeans_labels(feat, k=2):
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feat)


def gmm_labels(feat, k=2):
    return GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit_predict(feat)


def _check(n_images):
    from isic_utils import dice, features, labels_to_mask, load_image, load_split

    methods = {
        "Otsu": lambda rgb, X: otsu_labels(rgb),
        "K-Means": lambda rgb, X: kmeans_labels(X),
        "GMM": lambda rgb, X: gmm_labels(X),
    }
    scores = {name: [] for name in methods}
    for image_id in load_split("tune")[:n_images]:
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        row = []
        for name, fn in methods.items():
            scores[name].append(dice(labels_to_mask(fn(rgb, X), rgb), gt))
            row.append(f"{name} {scores[name][-1]:.3f}")
        print(f"{image_id}  " + "  ".join(row))
    print("Dice trung bình: " + "  ".join(f"{name} {np.mean(v):.3f}" for name, v in scores.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh baseline trên vài ảnh tune")
    parser.add_argument("--n", type=int, default=10, help="Số ảnh tune (mặc định 10)")
    _check(parser.parse_args().n)
