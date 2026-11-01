import inspect

from meanshift import (
    BruteForceSearch,
    FlatKernel,
    GaussianKernel,
    KDTreeSearch,
    MeanShiftCustom,
    estimate_bandwidth,
    get_bin_seeds,
)
from meanshift.base import BaseClusterer
from meanshift.kernels import make_kernel
from meanshift.neighbors import make_neighbor_search


def params(func):
    return {
        name: (None if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(func).parameters.items()
        if name != "self"
    }


def test_kernels_interface():
    for cls in (FlatKernel, GaussianKernel):
        assert list(params(cls.support)) == ["bandwidth"]
        assert list(params(cls.weights)) == ["sq_dist", "bandwidth"]
    assert params(GaussianKernel.__init__) == {"cutoff": 3.0}
    assert params(make_kernel) == {"kernel": None, "gaussian_cutoff": 3.0}


def test_neighbors_interface():
    for cls in (BruteForceSearch, KDTreeSearch):
        assert list(params(cls.fit)) == ["X"]
        assert list(params(cls.query_radius)) == ["points", "r"]
    assert params(BruteForceSearch.__init__) == {"chunk_size": 512}
    assert list(params(make_neighbor_search)) == ["name"]


def test_bandwidth_and_seeding_interface():
    assert params(estimate_bandwidth) == {
        "X": None, "quantile": 0.3, "n_samples": None, "random_state": 0,
    }
    assert params(get_bin_seeds) == {"X": None, "bin_size": None, "min_bin_freq": 1}


def test_meanshift_interface():
    assert issubclass(MeanShiftCustom, BaseClusterer)
    assert params(MeanShiftCustom.__init__) == {
        "bandwidth": None, "kernel": "flat", "neighbor_search": "kdtree",
        "gaussian_cutoff": 3.0, "seeds": None, "bin_seeding": False,
        "min_bin_freq": 1, "cluster_all": True, "max_iter": 300, "tol": 1e-3,
        "merge_radius": None, "record_trajectories": False,
    }
    assert list(params(MeanShiftCustom.fit)) == ["X", "y"]
    assert list(params(MeanShiftCustom.predict)) == ["X"]
    assert callable(MeanShiftCustom.fit_predict)
    assert MeanShiftCustom(bandwidth=2.0).get_params()["bandwidth"] == 2.0
