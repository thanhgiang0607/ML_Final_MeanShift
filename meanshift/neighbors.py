from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree

class NeighborSearch(ABC):
    name = "base"
    def fit(self, X):
        self.X_ = X
        return self
    @abstractmethod
    def query_radius(self, points, r):
        pass


class BruteForceSearch(NeighborSearch):

    name = "brute"

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def query_radius(self, points, r):
        X = np.asarray(self.X_, dtype=float)
        P = np.atleast_2d(np.asarray(points, dtype=float))
        r2 = float(r) ** 2
        x_sq = np.einsum("ij,ij->i", X, X)  
        step = max(1, int(self.chunk_size))
        result = []
        for start in range(0, len(P), step):
            chunk = P[start:start + step]
            p_sq = np.einsum("ij,ij->i", chunk, chunk)
            d2 = p_sq[:, None] - 2.0 * (chunk @ X.T) + x_sq[None, :]
            np.maximum(d2, 0.0, out=d2)
            for i in range(len(chunk)):
                idx = np.nonzero(d2[i] <= r2 * (1 + 1e-9) + 1e-12)[0]
                diff = X[idx] - chunk[i]
                sq = np.einsum("ij,ij->i", diff, diff)
                keep = sq <= r2
                result.append((idx[keep], sq[keep]))
        return result

class KDTreeSearch(NeighborSearch):

    name = "kdtree"

    def fit(self, X):
        super().fit(X)
        self.tree_ = cKDTree(np.asarray(X, dtype=float))
        return self

    def query_radius(self, points, r):
        X = np.asarray(self.X_, dtype=float)
        P = np.atleast_2d(np.asarray(points, dtype=float))
        r2 = float(r) ** 2
        neighbors = self.tree_.query_ball_point(P, r * (1 + 1e-9))
        result = []
        for i, ids in enumerate(neighbors):
            idx = np.sort(np.asarray(ids, dtype=np.intp))
            diff = X[idx] - P[i]
            sq = np.einsum("ij,ij->i", diff, diff)
            keep = sq <= r2
            result.append((idx[keep], sq[keep]))
        return result


def make_neighbor_search(name):
    if isinstance(name, NeighborSearch):
        return name
    if name == "brute":
        return BruteForceSearch()
    if name == "kdtree":
        return KDTreeSearch()
    raise ValueError(f"Tên bộ tìm láng giềng không hợp lệ: {name!r} (chỉ chấp nhận 'brute' hoặc 'kdtree')")
