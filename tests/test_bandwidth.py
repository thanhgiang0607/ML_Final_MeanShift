from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import estimate_bandwidth as sk_estimate_bandwidth
from sklearn.datasets import make_blobs

from meanshift.bandwidth import estimate_bandwidth

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
NAMES = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]


def load_synthetic(name, seed):
    path = SYNTHETIC / f"{name}_seed{seed}.csv"
    if not path.exists():
        pytest.skip("Chưa có dữ liệu: chạy `python3 data/make_synthetic.py`")
    return np.loadtxt(path, delimiter=",", skiprows=1)[:, :2]


@pytest.mark.parametrize("quantile", [0.1, 0.3, 0.5])
def test_matches_sklearn(quantile):
    X, _ = make_blobs(n_samples=500, centers=4, random_state=5)
    assert estimate_bandwidth(X, quantile=quantile) == pytest.approx(
        sk_estimate_bandwidth(X, quantile=quantile)
    )


def test_matches_sklearn_with_subsample():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=5)
    assert estimate_bandwidth(X, n_samples=200, random_state=1) == pytest.approx(
        sk_estimate_bandwidth(X, n_samples=200, random_state=1)
    )


def test_invalid_quantile():
    with pytest.raises(ValueError):
        estimate_bandwidth([[0.0, 0.0], [1.0, 1.0]], quantile=1.5)


@pytest.mark.parametrize("quantile", [-0.1, 1.01])
def test_invalid_quantile_both_sides(quantile):
    with pytest.raises(ValueError):
        estimate_bandwidth([[0.0, 0.0], [1.0, 1.0]], quantile=quantile)


@pytest.mark.parametrize("quantile", [0.0, 1.0])
def test_quantile_edges_match_sklearn(quantile):
    X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
    assert estimate_bandwidth(X, quantile=quantile) == pytest.approx(
        sk_estimate_bandwidth(X, quantile=quantile)
    )


def test_returns_float():
    X, _ = make_blobs(n_samples=100, centers=2, random_state=0)
    assert isinstance(estimate_bandwidth(X), float)


def test_k_is_at_least_one():
    X = np.random.RandomState(0).randn(10, 2)
    assert estimate_bandwidth(X, quantile=0.01) == 0.0


def test_accepts_list_input():
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]]
    assert estimate_bandwidth(X, quantile=0.5) == pytest.approx(
        sk_estimate_bandwidth(np.array(X), quantile=0.5)
    )


@pytest.mark.parametrize("random_state", [0, 1, 42])
def test_subsample_matches_sklearn_other_seeds(random_state):
    X, _ = make_blobs(n_samples=500, centers=4, random_state=3)
    assert estimate_bandwidth(X, n_samples=100, random_state=random_state) == pytest.approx(
        sk_estimate_bandwidth(X, n_samples=100, random_state=random_state)
    )


def test_subsample_is_reproducible():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=3)
    a = estimate_bandwidth(X, n_samples=100, random_state=7)
    b = estimate_bandwidth(X, n_samples=100, random_state=7)
    assert a == b


def test_n_samples_larger_than_n_matches_sklearn():
    X, _ = make_blobs(n_samples=50, centers=2, random_state=0)
    assert estimate_bandwidth(X, n_samples=500, random_state=1) == pytest.approx(
        sk_estimate_bandwidth(X, n_samples=500, random_state=1)
    )


@pytest.mark.parametrize("bad", [0, -5])
def test_n_samples_must_be_positive(bad):
    X, _ = make_blobs(n_samples=50, centers=2, random_state=0)
    with pytest.raises(ValueError):
        estimate_bandwidth(X, n_samples=bad)


def test_single_point_matches_sklearn():
    X = np.array([[1.0, 2.0]])
    assert estimate_bandwidth(X) == pytest.approx(sk_estimate_bandwidth(X)) == 0.0


def test_duplicate_points_match_sklearn():
    X = np.array([[1.0, 1.0]] * 5 + [[4.0, 4.0]])
    assert estimate_bandwidth(X, quantile=0.5) == pytest.approx(
        sk_estimate_bandwidth(X, quantile=0.5)
    )


def test_all_identical_points_give_zero():
    assert estimate_bandwidth(np.ones((20, 2))) == 0.0


@pytest.mark.parametrize("d", [1, 3, 5])
def test_other_dimensions_match_sklearn(d):
    X = np.random.RandomState(0).randn(120, d)
    assert estimate_bandwidth(X) == pytest.approx(sk_estimate_bandwidth(X))


def test_list_input_equals_array_input():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=2)
    assert estimate_bandwidth(X.tolist(), quantile=0.2) == estimate_bandwidth(X, quantile=0.2)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("name", NAMES)
def test_synthetic_matches_sklearn(name, seed):
    X = load_synthetic(name, seed)
    assert estimate_bandwidth(X) == pytest.approx(sk_estimate_bandwidth(X))


@pytest.mark.parametrize("quantile", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("name", NAMES)
def test_synthetic_quantiles_match_sklearn(name, quantile):
    X = load_synthetic(name, 0)
    assert estimate_bandwidth(X, quantile=quantile) == pytest.approx(
        sk_estimate_bandwidth(X, quantile=quantile)
    )


@pytest.mark.parametrize("name", NAMES)
def test_synthetic_subsample_matches_sklearn(name):
    X = load_synthetic(name, 0)
    assert estimate_bandwidth(X, n_samples=300, random_state=1) == pytest.approx(
        sk_estimate_bandwidth(X, n_samples=300, random_state=1)
    )
