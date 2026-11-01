from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    name = "base"

    @abstractmethod
    def support(self, bandwidth):
        pass

    @abstractmethod
    def weights(self, sq_dist, bandwidth):
        pass


class FlatKernel(Kernel):

    name = "flat"

    def support(self, bandwidth):
        return bandwidth

    def weights(self, sq_dist, bandwidth):
        return np.ones_like(np.asarray(sq_dist, dtype=np.float64))


class GaussianKernel(Kernel):

    name = "gaussian"

    def __init__(self, cutoff=3.0):
        if not cutoff > 0:
            raise ValueError(f"cutoff phải > 0, nhận được {cutoff!r}.")
        self.cutoff = cutoff

    def support(self, bandwidth):
        return self.cutoff * bandwidth

    def weights(self, sq_dist, bandwidth):
        sq_dist = np.asarray(sq_dist, dtype=np.float64)
        return np.exp(-0.5 * sq_dist / bandwidth**2)


def make_kernel(kernel, gaussian_cutoff=3.0):
    if isinstance(kernel, Kernel):
        return kernel
    if kernel == "flat":
        return FlatKernel()
    if kernel == "gaussian":
        return GaussianKernel(cutoff=gaussian_cutoff)
    raise ValueError(f"Kernel không hợp lệ: {kernel!r}. Chọn 'flat' hoặc 'gaussian'.")
