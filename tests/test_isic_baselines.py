import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from isic_baselines import gmm_labels, kmeans_labels, otsu_labels
from isic_utils import features, labels_to_mask


def disk_image(h=60, w=80, radius=15, seed=0):
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= radius ** 2
    rgb = np.where(disk[..., None], (90, 50, 30), (230, 200, 180)).astype(float)
    rgb += np.random.default_rng(seed).normal(0, 4, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8), disk


@pytest.fixture
def image():
    rgb, disk = disk_image()
    return rgb, disk, features(rgb, "lab3")


def test_otsu_separates_dark_disk(image):
    rgb, disk, _ = image
    labels = otsu_labels(rgb)
    assert labels.shape == (disk.size,)
    assert np.mean((labels == 0) == disk.ravel()) > 0.999


def test_otsu_constant_image():
    rgb = np.full((10, 12, 3), 128, dtype=np.uint8)
    np.testing.assert_array_equal(otsu_labels(rgb), np.zeros(120, dtype=int))


@pytest.mark.parametrize("fn", [kmeans_labels, gmm_labels])
def test_clustering_shape_and_k(image, fn):
    _, disk, X = image
    labels = fn(X)
    assert labels.shape == (disk.size,)
    assert set(np.unique(labels)) == {0, 1}
    assert set(np.unique(fn(X, k=3))) <= {0, 1, 2}


@pytest.mark.parametrize("fn", [kmeans_labels, gmm_labels])
def test_clustering_is_deterministic(image, fn):
    _, _, X = image
    np.testing.assert_array_equal(fn(X), fn(X))


@pytest.mark.parametrize("make_labels", [
    lambda rgb, X: otsu_labels(rgb),
    lambda rgb, X: kmeans_labels(X),
    lambda rgb, X: gmm_labels(X),
])
def test_labels_feed_into_labels_to_mask(image, make_labels):
    rgb, disk, X = image
    np.testing.assert_array_equal(labels_to_mask(make_labels(rgb, X), rgb), disk)
