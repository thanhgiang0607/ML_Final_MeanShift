import inspect

import numpy as np


class BaseClusterer:

    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self):
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        valid = self._param_names()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Tham số không hợp lệ: {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X):
        raise NotImplementedError

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _check_is_fitted(self):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError(f"{type(self).__name__} chưa được fit.")

    def __repr__(self):
        defaults = {
            name: p.default
            for name, p in inspect.signature(type(self).__init__).parameters.items()
            if name != "self"
        }
        changed = [
            f"{k}={v!r}" for k, v in self.get_params().items()
            if not (isinstance(v, np.ndarray) or v == defaults[k])
        ]
        return f"{type(self).__name__}({', '.join(changed)})"


def check_array(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X phải là mảng 2D, nhận được ndim={X.ndim}.")
    if X.shape[0] == 0:
        raise ValueError("X rỗng.")
    if not np.isfinite(X).all():
        raise ValueError("X chứa NaN hoặc inf.")
    return X
