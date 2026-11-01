import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from meanshift.neighbors import BruteForceSearch, KDTreeSearch, make_neighbor_search


def as_sets(result):
    return [set(idx.tolist()) for idx, _ in result]


def test_brute_and_kdtree_find_same_neighbors():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    Q = rng.randn(20, 2)
    a = BruteForceSearch().fit(X).query_radius(Q, 0.7)
    b = KDTreeSearch().fit(X).query_radius(Q, 0.7)
    assert as_sets(a) == as_sets(b)


def test_sq_dist_is_squared_distance():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    for search in (BruteForceSearch(), KDTreeSearch()):
        (idx, sq), = search.fit(X).query_radius(np.array([[0.0, 0.0]]), 10.0)
        order = np.argsort(idx)
        np.testing.assert_allclose(sq[order], [0.0, 25.0], atol=1e-9)


def test_point_on_boundary_is_included():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    for search in (BruteForceSearch(), KDTreeSearch()):
        (idx, _), = search.fit(X).query_radius(np.array([[0.0, 0.0]]), 1.0)
        assert set(idx.tolist()) == {0, 1}


def test_make_neighbor_search():
    assert isinstance(make_neighbor_search("brute"), BruteForceSearch)
    assert isinstance(make_neighbor_search("kdtree"), KDTreeSearch)
    with pytest.raises(ValueError):
        make_neighbor_search("abc")

@pytest.mark.parametrize("search_cls", [BruteForceSearch, KDTreeSearch])
def test_diem_tren_bien_voi_r_bang_dung_khoang_cach(search_cls):
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [10.0, 0.0]])
    q = np.array([[0.0, 0.0]])

    r = float(np.linalg.norm(X[1] - q[0]))
    assert r == 5.0

    (idx, sq), = search_cls().fit(X).query_radius(q, r)
    assert 1 in idx.tolist()
    assert set(idx.tolist()) == {0, 1}
    assert_allclose(sq[np.argsort(idx)], [0.0, 25.0])

    (idx2, _), = search_cls().fit(X).query_radius(q, r - 1e-6)
    assert set(idx2.tolist()) == {0}


@pytest.mark.parametrize("chunk_size", [1, 7, 512])
@pytest.mark.parametrize("r", [0.1, 0.4, 0.8, 1.5])
def test_brute_va_kdtree_khop_idx_va_sq_dist(r, chunk_size):
    rng = np.random.RandomState(42)
    X = rng.randn(400, 3)
    Q = rng.randn(50, 3)

    brute = BruteForceSearch(chunk_size=chunk_size).fit(X).query_radius(Q, r)
    kd = KDTreeSearch().fit(X).query_radius(Q, r)

    assert len(brute) == len(kd) == len(Q)
    for (bi, bs), (ki, ks) in zip(brute, kd):
        assert_array_equal(bi, ki)
        assert_allclose(bs, ks, rtol=1e-12, atol=1e-12)


def test_chunk_size_nho_cho_cung_ket_qua_voi_chunk_lon():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    Q = rng.randn(33, 2)
    nho = BruteForceSearch(chunk_size=4).fit(X).query_radius(Q, 0.6)
    lon = BruteForceSearch(chunk_size=10_000).fit(X).query_radius(Q, 0.6)
    for (ai, asq), (bi, bsq) in zip(nho, lon):
        assert_array_equal(ai, bi)
        assert_allclose(asq, bsq)


@pytest.mark.parametrize("ten_la", ["abc", "", "KDTREE", None])
def test_ten_la_bao_value_error(ten_la):
    with pytest.raises(ValueError):
        make_neighbor_search(ten_la)

def test_diem_trung_toa_do():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [2.0, 2.0],
        [1.0, 1.0],
    ])
    points = np.array([[0.0, 0.0], [1.0, 1.0]])

    expected = [
        (np.array([0, 2]), np.array([0.0, 0.0])),
        (np.array([1, 4]), np.array([0.0, 0.0])),
    ]

    for search in [
        BruteForceSearch(chunk_size=1),
        KDTreeSearch(),
    ]:
        search.fit(X)
        results = search.query_radius(points, r=0.5)

        assert len(results) == len(expected)

        for (idx, sq_dist), (expected_idx, expected_sq) in zip(
            results, expected
        ):
            np.testing.assert_array_equal(idx, expected_idx)
            assert_allclose(sq_dist, expected_sq, atol=1e-12)


def test_ban_kinh_bang_0():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [2.0, 2.0],
    ])
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [5.0, 5.0],
    ])

    expected_indices = [
        np.array([0, 2]),
        np.array([1]),
        np.array([], dtype=int),
    ]
    expected_sq_distances = [
        np.array([0.0, 0.0]),
        np.array([0.0]),
        np.array([]),
    ]

    for search in [
        BruteForceSearch(chunk_size=1),
        KDTreeSearch(),
    ]:
        search.fit(X)
        results = search.query_radius(points, r=0.0)

        assert len(results) == len(points)

        for (idx, sq_dist), expected_idx, expected_sq in zip(
            results, expected_indices, expected_sq_distances
        ):
            assert idx.dtype.kind == "i"
            np.testing.assert_array_equal(idx, expected_idx)
            assert_allclose(sq_dist, expected_sq, atol=1e-12)
