from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any

import numpy as np

__all__ = ["StandardScaler", "PolynomialFeatures"]


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
        """Map (W, b) fit in standardised space back to original feature space."""
        W_raw = W / self.std_  # (n, k), broadcast over columns
        b_raw = b - (W_raw * self.mean_).sum(axis=0)  # (k,)
        return W_raw, b_raw


class PolynomialFeatures:
    """Generate all monomial feature combinations up to degree."""

    def __init__(self, degree: int = 2, include_bias: bool = False) -> None:
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be an integer >= 1.")
        self.degree = degree
        self.include_bias = include_bias
        self.n_features_in_: int | None = None
        self.n_output_features_: int | None = None

    def _combos(self, n_features: int) -> list[tuple[int, ...]]:
        combos: list[tuple[int, ...]] = []
        for deg in range(0, self.degree + 1):
            if deg == 0 and not self.include_bias:
                continue
            combos.extend(combinations_with_replacement(range(n_features), deg))
        return combos

    def fit(self, X: Any, y: Any = None) -> "PolynomialFeatures":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.n_output_features_ = len(self._combos(self.n_features_in_))
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n = X.shape
        cols: list[np.ndarray] = []
        for deg in range(0, self.degree + 1):
            if deg == 0 and not self.include_bias:
                continue
            for combo in combinations_with_replacement(range(n), deg):
                cols.append(
                    np.ones(n_samples) if deg == 0 else np.prod(X[:, combo], axis=1)
                )
        if not cols:
            cols.append(np.ones(n_samples))
        return np.column_stack(cols)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X).transform(X)
