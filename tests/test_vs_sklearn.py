import numpy as np
import pytest
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from meanshift import MeanShiftCustom


@pytest.mark.parametrize("bin_seeding", [False, True])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_flat_kernel_matches_sklearn_on_blobs(bin_seeding, search):
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=7)
    h = 1.2
    ours = MeanShiftCustom(bandwidth=h, neighbor_search=search, bin_seeding=bin_seeding).fit(X)
    ref = MeanShift(bandwidth=h, bin_seeding=bin_seeding, n_jobs=1).fit(X)
    assert len(ours.cluster_centers_) == len(ref.cluster_centers_)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)


def test_nonconvex_moons_partition_parity_without_assuming_truth_recovery():
    X, _ = make_moons(n_samples=300, noise=0.04, random_state=13)
    ours = MeanShiftCustom(bandwidth=0.45, neighbor_search="kdtree").fit(X)
    ref = MeanShift(bandwidth=0.45, n_jobs=1).fit(X)
    assert len(ours.cluster_centers_) == len(ref.cluster_centers_)
    assert adjusted_rand_score(ours.labels_, ref.labels_) > 0.99


def test_explicit_seeds_define_same_modes_with_both_search_backends():
    X, _ = make_blobs(n_samples=180, centers=3, cluster_std=0.35, random_state=21)
    seeds = X[[0, 20, 40, 60, 80, 100]]
    outputs = [MeanShiftCustom(bandwidth=0.8, seeds=seeds, neighbor_search=backend).fit(X)
               for backend in ("brute", "kdtree")]
    assert len(outputs[0].cluster_centers_) == len(outputs[1].cluster_centers_)
    np.testing.assert_allclose(outputs[0].cluster_centers_, outputs[1].cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(outputs[0].labels_, outputs[1].labels_)


def test_cluster_all_false_keeps_orphans_but_predict_assigns_nearest():
    X = np.array([[0.0], [0.1], [-0.1], [10.0], [10.1], [9.9], [100.0]])
    seeds = np.array([[0.0], [10.0]])
    model = MeanShiftCustom(bandwidth=0.5, seeds=seeds, cluster_all=False).fit(X)
    reference = MeanShift(bandwidth=0.5, seeds=seeds, cluster_all=False, n_jobs=1).fit(X)
    np.testing.assert_array_equal(model.labels_, reference.labels_)
    assert model.labels_[-1] == -1
    predicted = model.predict(X)
    assert predicted[-1] >= 0
    np.testing.assert_array_equal(predicted, reference.predict(X))
    np.testing.assert_array_equal(predicted[:-1], model.labels_[:-1])


def test_tied_modes_merge_and_iteration_count_obeys_limit():
    X = np.repeat(np.array([[0.0, 0.0], [3.0, 0.0]]), [8, 8], axis=0)
    model = MeanShiftCustom(bandwidth=0.4, max_iter=2).fit(X)
    assert len(model.cluster_centers_) == 2
    assert model.n_iter_ <= 2
    np.testing.assert_allclose(np.sort(model.cluster_centers_[:, 0]), [0.0, 3.0], atol=1e-10)


def test_gaussian_trajectory_is_cutoff_approximation_to_full_kde_update():
    X = np.array([[0.0], [0.2], [0.4], [1.0]])
    h = 0.3
    model = MeanShiftCustom(bandwidth=h, kernel="gaussian", gaussian_cutoff=2.0,
                            record_trajectories=True, seeds=np.array([[0.0]])).fit(X)
    scaled_sq = (X[:, 0] / h) ** 2
    full_gaussian_update = np.average(X[:, 0], weights=np.exp(-0.5 * scaled_sq))
    truncated_update = model.trajectories_[0][1, 0]
    inside = scaled_sq <= 2.0**2
    expected = np.average(X[inside, 0], weights=np.exp(-0.5 * scaled_sq[inside]))
    assert truncated_update == pytest.approx(expected)
    assert truncated_update != pytest.approx(full_gaussian_update)
