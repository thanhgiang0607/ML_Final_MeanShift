import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import isic_utils
from isic_utils import border_ring, dice, features, iou, labels_to_mask, load_image, load_split, to_lab


def disk_image(h=60, w=80, radius=15):
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= radius ** 2
    rgb = np.full((h, w, 3), (230, 200, 180), dtype=np.uint8)
    rgb[disk] = (90, 50, 30)
    return rgb, disk


def test_dice_iou_values():
    pred = np.array([[1, 1, 0, 0]], dtype=bool)
    gt = np.array([[1, 0, 0, 0]], dtype=bool)
    assert dice(pred, gt) == pytest.approx(2 / 3)
    assert iou(pred, gt) == pytest.approx(1 / 2)
    assert dice(gt, gt) == iou(gt, gt) == 1.0
    assert dice(pred, ~pred) == iou(pred, ~pred) == 0.0


def test_dice_iou_both_empty():
    empty = np.zeros((3, 3), dtype=bool)
    assert dice(empty, empty) == iou(empty, empty) == 1.0


def test_features_shapes_and_units():
    rgb, _ = disk_image()
    n = rgb.shape[0] * rgb.shape[1]
    assert features(rgb, "rgb3").shape == (n, 3)
    lab = features(rgb, "lab3")
    assert lab.shape == (n, 3)
    assert 0 <= lab[:, 0].min() and lab[:, 0].max() <= 100
    np.testing.assert_allclose(lab, to_lab(rgb).reshape(-1, 3))


def test_features_lab5_scaling():
    rgb, _ = disk_image(h=4, w=5)
    X = features(rgb, "lab5", hs=2, hr=4)
    assert X.shape == (20, 5)
    np.testing.assert_allclose(X[7, :2], [2 / 2, 1 / 2])
    np.testing.assert_allclose(X[:, 2:], to_lab(rgb).reshape(-1, 3) / 4)


@pytest.mark.parametrize("kwargs", [{}, {"hs": 8}, {"hr": 8}, {"hs": 0, "hr": 8}])
def test_features_lab5_needs_positive_scales(kwargs):
    rgb, _ = disk_image()
    with pytest.raises(ValueError):
        features(rgb, "lab5", **kwargs)


def test_features_unknown_mode():
    rgb, _ = disk_image()
    with pytest.raises(ValueError):
        features(rgb, "hsv3")


def test_border_ring_width():
    ring = border_ring((40, 60), width=0.05)
    assert ring[:2].all() and ring[-2:].all() and ring[:, :2].all() and ring[:, -2:].all()
    assert not ring[2:-2, 2:-2].any()


def test_labels_to_mask_picks_dark_center():
    rgb, disk = disk_image()
    labels = disk.astype(int).ravel()
    np.testing.assert_array_equal(labels_to_mask(labels, rgb), disk)


def test_labels_to_mask_label_values_do_not_matter():
    rgb, disk = disk_image()
    labels = np.where(disk, 7, 3).ravel()
    np.testing.assert_array_equal(labels_to_mask(labels, rgb), disk)


def test_labels_to_mask_skips_dark_border_and_tiny_clusters():
    rgb, disk = disk_image()
    labels = disk.astype(int)
    rgb[:3, :3] = (0, 0, 0)
    labels[:3, :3] = 2
    rgb[10, 10] = (0, 0, 0)
    labels[10, 10] = 3
    np.testing.assert_array_equal(labels_to_mask(labels.ravel(), rgb), disk)


def test_labels_to_mask_fills_holes_and_keeps_largest_component():
    rgb, disk = disk_image()
    labels = disk.astype(int)
    labels[30, 40] = 0
    labels[50:53, 70:73] = 1
    rgb[50:53, 70:73] = (90, 50, 30)
    mask = labels_to_mask(labels.ravel(), rgb)
    np.testing.assert_array_equal(mask, disk)


def test_labels_to_mask_ignores_noise_label():
    rgb, disk = disk_image()
    labels = np.where(disk, -1, 0).ravel()
    assert not labels_to_mask(labels, rgb).any()


def test_labels_to_mask_single_cluster_gives_empty_mask():
    rgb, _ = disk_image()
    assert not labels_to_mask(np.zeros(rgb.shape[0] * rgb.shape[1], dtype=int), rgb).any()


def test_load_split_rejects_unknown_name():
    with pytest.raises(ValueError):
        load_split("train")


def test_splits_sizes():
    assert len(load_split("tune")) == 100
    assert len(load_split("test")) == 200
    assert not set(load_split("tune")) & set(load_split("test"))


def test_load_image_real():
    image_id = load_split("tune")[0]
    if not (isic_utils.IMAGE_DIRS[0] / f"{image_id}.jpg").is_file():
        pytest.skip("Chưa tải ISIC: chạy `python3 data/download_isic.py`")
    rgb, mask = load_image(image_id)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert max(rgb.shape[:2]) == 128
    assert mask.dtype == bool and mask.shape == rgb.shape[:2]
    assert 0 < mask.mean() < 1
