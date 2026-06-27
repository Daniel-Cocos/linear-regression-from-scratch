from __future__ import annotations

import numpy as np

from .base import BaseRegression
from .solvers import soft_threshold

__all__ = ["LinearRegression", "RidgeRegression", "LassoRegression"]


class LinearRegression(BaseRegression):
    pass  # no penalty; the base engine is already plain OLS


class RidgeRegression(BaseRegression):
    """Linear regression with an L2 penalty (Tikhonov / weight decay)"""

    def __init__(
        self, alpha: float = 1.0, lr: float = 0.1, n_iter: int = 2000, **kwargs
    ) -> None:
        kwargs.setdefault("solver", "batch")
        super().__init__(lr=lr, n_iter=n_iter, **kwargs)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)

    def _penalty_gradient(self, W: np.ndarray) -> np.ndarray:
        return self.alpha * W

    def _penalty_cost(self, W: np.ndarray) -> float:
        return 0.5 * self.alpha * float(np.sum(W**2))


class LassoRegression(BaseRegression):
    """Linear regression with an L1 penalty produces sparse weights"""

    def __init__(
        self, alpha: float = 1.0, lr: float = 0.5, n_iter: int = 2000, **kwargs
    ) -> None:
        kwargs.setdefault("solver", "batch")
        super().__init__(lr=lr, n_iter=n_iter, **kwargs)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)

    # The L1 term is handled by the proximal step, so the gradient carries only
    # the smooth squared-error data term (already provided by the base class)
    def _penalty_cost(self, W: np.ndarray) -> float:
        return self.alpha * float(np.sum(np.abs(W)))

    def _prox(self, W: np.ndarray) -> np.ndarray:
        return soft_threshold(W, self.lr * self.alpha)
