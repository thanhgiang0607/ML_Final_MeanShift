import numpy as np
import pytest

from meanshift.kernels import FlatKernel, GaussianKernel, make_kernel


def test_flat_weights_all_one():
    w = FlatKernel().weights(np.array([0.0, 0.5, 1.0]), 1.0)
    np.testing.assert_array_equal(w, [1.0, 1.0, 1.0])


def test_gaussian_weights_formula():
    h = 2.0
    sq = np.array([0.0, h**2, 4 * h**2])
    w = GaussianKernel().weights(sq, h)
    np.testing.assert_allclose(w, np.exp([-0.0, -0.5, -2.0]))


def test_support():
    assert FlatKernel().support(1.5) == 1.5
    assert GaussianKernel(cutoff=3.0).support(1.5) == pytest.approx(4.5)


def test_make_kernel():
    assert isinstance(make_kernel("flat"), FlatKernel)
    assert isinstance(make_kernel("gaussian"), GaussianKernel)
    with pytest.raises(ValueError):
        make_kernel("abc")


def test_flat_weights_shape_and_dtype():
    sq = np.array([0.0, 2.0, 5.0, 9.0])
    w = FlatKernel().weights(sq, 3.0)
    assert w.shape == sq.shape
    assert w.dtype == np.float64


def test_weights_accept_list():
    np.testing.assert_array_equal(FlatKernel().weights([0.0, 1.0], 1.0), [1.0, 1.0])
    np.testing.assert_allclose(GaussianKernel().weights([0.0], 1.0), [1.0])


def test_gaussian_weights_decrease_with_distance():
    w = GaussianKernel().weights(np.array([0.0, 1.0, 4.0, 9.0]), 1.0)
    assert w[0] == pytest.approx(1.0)
    assert np.all(np.diff(w) < 0)
    assert np.all(w > 0)


def test_gaussian_default_cutoff():
    assert GaussianKernel().cutoff == 3.0


@pytest.mark.parametrize("bad", [0, -1.0, float("nan")])
def test_gaussian_invalid_cutoff(bad):
    with pytest.raises(ValueError):
        GaussianKernel(cutoff=bad)


def test_make_kernel_passes_cutoff():
    k = make_kernel("gaussian", gaussian_cutoff=2.0)
    assert k.cutoff == 2.0
    assert k.support(1.0) == pytest.approx(2.0)


def test_make_kernel_returns_same_object():
    k = GaussianKernel(cutoff=2.5)
    assert make_kernel(k) is k


def test_empty_sq_dist():
    assert FlatKernel().weights(np.array([]), 1.0).shape == (0,)
    assert GaussianKernel().weights(np.array([]), 1.0).shape == (0,)
