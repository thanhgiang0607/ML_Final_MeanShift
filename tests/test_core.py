import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from meanshift import MeanShiftCustom, estimate_bandwidth


def sort_rows(a):
    return a[np.lexsort(a.T[::-1])]


@pytest.fixture
def two_blobs():
    return make_blobs(n_samples=300, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)


@pytest.mark.parametrize("kernel", ["flat", "gaussian"])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_two_far_blobs_give_two_clusters(two_blobs, kernel, search):
    X, y = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, kernel=kernel, neighbor_search=search).fit(X)
    assert len(ms.cluster_centers_) == 2
    assert adjusted_rand_score(y, ms.labels_) == 1.0


def test_single_point():
    ms = MeanShiftCustom(bandwidth=1.0).fit([[3.0, 4.0]])
    np.testing.assert_allclose(ms.cluster_centers_, [[3.0, 4.0]])
    assert ms.labels_.tolist() == [0]


def test_predict_matches_labels(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    np.testing.assert_array_equal(ms.predict(X), ms.labels_)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="chưa được fit"):
        MeanShiftCustom().predict([[0.0, 0.0]])


def test_bandwidth_none_uses_estimate_bandwidth(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom().fit(X)
    assert ms.bandwidth_ == pytest.approx(estimate_bandwidth(X))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_bandwidth_must_be_positive(two_blobs, bad):
    X, _ = two_blobs
    with pytest.raises(ValueError):
        MeanShiftCustom(bandwidth=bad).fit(X)


def test_fit_returns_self_and_fit_predict_matches_labels(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0)
    assert ms.fit(X) is ms
    np.testing.assert_array_equal(MeanShiftCustom(bandwidth=2.0).fit_predict(X), ms.labels_)


def test_fitted_attributes_have_right_shape(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    k = len(ms.cluster_centers_)
    assert ms.cluster_centers_.shape == (k, X.shape[1])
    assert ms.labels_.shape == (len(X),)
    assert ms.intensities_.shape == (k,)
    assert ms.seed_n_iter_.shape == (len(X),)
    assert ms.n_iter_ == ms.seed_n_iter_.max()
    assert list(ms.intensities_) == sorted(ms.intensities_, reverse=True)


@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_bin_seeding_gives_two_clusters(two_blobs, search):
    X, y = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, bin_seeding=True, neighbor_search=search).fit(X)
    assert len(ms.cluster_centers_) == 2
    assert adjusted_rand_score(y, ms.labels_) == 1.0
    assert len(ms.seed_n_iter_) < len(X)


def test_user_seeds(two_blobs):
    X, y = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, seeds=[[1.0, 1.0], [9.0, 9.0]]).fit(X)
    assert len(ms.seed_n_iter_) == 2
    assert len(ms.cluster_centers_) == 2
    assert adjusted_rand_score(y, ms.labels_) == 1.0


def test_seeds_without_neighbors_raise(two_blobs):
    X, _ = two_blobs
    with pytest.raises(ValueError):
        MeanShiftCustom(bandwidth=1.0, seeds=[[100.0, 100.0]]).fit(X)


def test_cluster_all_false_marks_outlier():
    X = np.vstack([np.random.RandomState(0).normal(0, 0.3, size=(50, 2)), [[20.0, 20.0]]])
    seeds = [[0.0, 0.0]]
    ms = MeanShiftCustom(bandwidth=2.0, seeds=seeds, cluster_all=False).fit(X)
    assert ms.labels_[-1] == -1
    assert (ms.labels_[:-1] == 0).all()
    ms_all = MeanShiftCustom(bandwidth=2.0, seeds=seeds, cluster_all=True).fit(X)
    assert (ms_all.labels_ == 0).all()


def test_max_iter_limits_iterations(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, max_iter=1).fit(X)
    assert ms.n_iter_ <= 1


def test_large_merge_radius_gives_one_cluster(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, merge_radius=100.0).fit(X)
    assert len(ms.cluster_centers_) == 1
    assert (ms.labels_ == 0).all()


def test_record_trajectories(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, record_trajectories=True).fit(X)
    assert len(ms.trajectories_) == len(X)
    for i, traj in enumerate(ms.trajectories_[:10]):
        np.testing.assert_allclose(traj[0], X[i])
        assert traj.shape[1] == X.shape[1]
    assert not hasattr(MeanShiftCustom(bandwidth=2.0).fit(X), "trajectories_")


def test_predict_new_points(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    near0 = ms.predict([[0.2, -0.1]])[0]
    near1 = ms.predict([[9.8, 10.1]])[0]
    assert near0 != near1
    np.testing.assert_allclose(ms.cluster_centers_[near0], [0, 0], atol=0.3)
    np.testing.assert_allclose(ms.cluster_centers_[near1], [10, 10], atol=0.3)


def test_get_set_params():
    ms = MeanShiftCustom(bandwidth=1.5, kernel="gaussian")
    params = ms.get_params()
    assert params["bandwidth"] == 1.5 and params["kernel"] == "gaussian"
    ms.set_params(bandwidth=3.0)
    assert ms.bandwidth == 3.0


def test_gaussian_close_to_flat_on_far_blobs(two_blobs):
    X, _ = two_blobs
    flat = MeanShiftCustom(bandwidth=2.0, kernel="flat").fit(X)
    gauss = MeanShiftCustom(bandwidth=2.0, kernel="gaussian").fit(X)
    np.testing.assert_allclose(sort_rows(flat.cluster_centers_), sort_rows(gauss.cluster_centers_), atol=0.2)


def test_3d_data():
    X, y = make_blobs(n_samples=300, centers=[[0, 0, 0], [8, 8, 8], [0, 8, 0]],
                      cluster_std=0.5, random_state=1)
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    assert len(ms.cluster_centers_) == 3
    assert adjusted_rand_score(y, ms.labels_) == 1.0
