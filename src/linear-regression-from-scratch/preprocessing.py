from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["StandardScaler"]


class StandardScaler:
    """Standardise features to zero mean and unit (population) variance."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0.0] = 1.0  # guard constant columns
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X * self.std_ + self.mean_

    def unstandardize_weights(
        self, W: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map (W, b) fit in standardised space back to original feature space.

        For ``z = (x - mu) / sigma`` and ``y ~= W z + b``::

            y = (W / sigma) . x + (b - sum(W * mu / sigma))

        Works for both single-output (W is (n, 1)) and multi-output (W is (n, k)).
        """
        W_raw = W / self.std_                       # (n, k), broadcast over columns
        b_raw = b - (W_raw * self.mean_).sum(axis=0)  # (k,)
        return W_raw, b_raw
