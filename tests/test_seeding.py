import warnings

import numpy as np
import pytest
from sklearn.cluster import estimate_bandwidth as sk_estimate_bandwidth
from sklearn.cluster import get_bin_seeds as sk_get_bin_seeds
from sklearn.datasets import make_blobs
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from meanshift.seeding import get_bin_seeds



def sort_rows(a):
    return a[np.lexsort(a.T[::-1])]


NAMES = ["iris", "wine", "breast_cancer"]
LOADERS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}


def load_real(name, seed):
    data = LOADERS[name]()
    X, y = data.data, data.target
    if seed > 0:
        X, _, y, _ = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)
    return PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))


@pytest.mark.parametrize("bin_size, min_bin_freq", [(1.0, 1), (0.5, 3)])
def test_matches_sklearn(bin_size, min_bin_freq):
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    ours = get_bin_seeds(X, bin_size, min_bin_freq)
    theirs = sk_get_bin_seeds(X, bin_size, min_bin_freq)
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


def test_bin_size_zero_returns_X():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(get_bin_seeds(X, 0), X)


@pytest.mark.parametrize("min_bin_freq", [1, 3, 10])
@pytest.mark.parametrize("bin_size", [0.5, 1.0, 2.0])
def test_matches_sklearn_grid(bin_size, min_bin_freq):
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=0)
    ours = get_bin_seeds(X, bin_size, min_bin_freq)
    theirs = sk_get_bin_seeds(X, bin_size, min_bin_freq)
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


def test_seeds_are_float32_cells_times_bin_size():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    ours = get_bin_seeds(X, 1.0)
    theirs = sk_get_bin_seeds(X, 1.0)
    assert ours.dtype == theirs.dtype == np.float32


def test_min_bin_freq_filters_sparse_cells():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, -0.1],
                  [5.0, 5.0]])
    np.testing.assert_allclose(get_bin_seeds(X, 1.0, min_bin_freq=2), [[0.0, 0.0]])
    np.testing.assert_allclose(
        sort_rows(get_bin_seeds(X, 1.0, min_bin_freq=1)), [[0.0, 0.0], [5.0, 5.0]]
    )


def test_seeds_are_multiples_of_bin_size():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    seeds = get_bin_seeds(X, 1.5)
    np.testing.assert_allclose(seeds / 1.5, np.round(seeds / 1.5), atol=1e-5)


def test_warns_and_returns_X_when_binning_does_not_reduce():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=0)
    with pytest.warns(UserWarning, match="Binning data failed"):
        out = get_bin_seeds(X, 1e-6)
    np.testing.assert_array_equal(out, X)


def test_no_warning_when_binning_reduces():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_bin_seeds(X, 1.0)


def test_min_bin_freq_too_large_gives_empty_seeds():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=0)
    ours = get_bin_seeds(X, 1.0, min_bin_freq=10_000)
    theirs = sk_get_bin_seeds(X, 1.0, min_bin_freq=10_000)
    assert ours.size == theirs.size == 0
    assert ours.dtype == theirs.dtype == np.float32
    assert ours.shape == (0, X.shape[1])


def test_single_point_matches_sklearn():
    X = np.array([[1.0, 2.0]])
    with pytest.warns(UserWarning):
        ours = get_bin_seeds(X, 1.0)
    with pytest.warns(UserWarning):
        theirs = sk_get_bin_seeds(X, 1.0)
    np.testing.assert_array_equal(ours, theirs)


def test_duplicate_points_collapse_to_one_cell():
    X = np.array([[1.0, 1.0]] * 5 + [[4.0, 4.0]])
    ours = get_bin_seeds(X, 1.0)
    theirs = sk_get_bin_seeds(X, 1.0)
    assert ours.shape == theirs.shape == (2, 2)
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


def test_negative_and_tie_coordinates_match_sklearn():
    X = np.array([[0.5, 0.5], [1.5, 1.5], [-0.5, -0.5], [2.5, -1.5], [-2.5, 2.5], [9.0, 9.0]])
    np.testing.assert_allclose(
        sort_rows(get_bin_seeds(X, 1.0)), sort_rows(sk_get_bin_seeds(X, 1.0))
    )


@pytest.mark.parametrize("d", [1, 3, 5])
def test_other_dimensions_match_sklearn(d):
    X = np.random.RandomState(0).randn(200, d)
    ours = get_bin_seeds(X, 1.0)
    theirs = sk_get_bin_seeds(X, 1.0)
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


def test_list_input_equals_array_input():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=2)
    np.testing.assert_array_equal(get_bin_seeds(X.tolist(), 1.0), get_bin_seeds(X, 1.0))


@pytest.mark.parametrize("min_bin_freq", [1, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", NAMES)
def test_real_matches_sklearn(name, seed, min_bin_freq):
    X = load_real(name, seed)
    h = sk_estimate_bandwidth(X)
    ours = get_bin_seeds(X, h, min_bin_freq)
    theirs = sk_get_bin_seeds(X, h, min_bin_freq)
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


@pytest.mark.parametrize("name", NAMES)
def test_real_has_fewer_seeds_than_points(name):
    X = load_real(name, 0)
    seeds = get_bin_seeds(X, sk_estimate_bandwidth(X))
    assert 0 < len(seeds) < len(X)
