from __future__ import annotations

import logging

import numpy as np

from .preprocessing import StandardScaler
from .solvers import SOLVERS
from .utils import (
    check_non_negative,
    check_positive,
)

_ALL_SOLVERS = (*SOLVERS, "normal")

class BaseRegression:
    """Shared linear-regression engine. Use a concrete subclass instead."""

    def __init__(
        self,
        *,
        lr: float = 0.1,
        n_iter: int = 1000,
        solver: str = "batch",
        tol: float = 1e-8,
        batch_size: int = 32,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        standardize: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if solver not in _ALL_SOLVERS:
            raise ValueError(f"solver must be one of {_ALL_SOLVERS}, got {solver!r}.")
        check_positive(lr, "lr")
        check_positive(n_iter, "n_iter")
        check_non_negative(tol, "tol")
        self.lr = lr
        self.n_iter = int(n_iter)
        self.solver = solver
        self.tol = tol
        self.batch_size = int(batch_size)
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Learned attributes (populated by ``fit``).
        self.w_: np.ndarray | None = None
        self.b_: float | np.ndarray | None = None
        self.cost_history_: list[float] = []
        self.n_iter_: int = 0
        self.scaler_: StandardScaler | None = None
        self.n_features_in_: int | None = None

    # Penalty hooks -- overridden by subclasses
    def _penalty_gradient(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)

    def _penalty_cost(self, W: np.ndarray) -> float:
        return 0.0

    def _prox(self, W: np.ndarray) -> np.ndarray:
        return W

    # Gradients & cost on the (smooth) data term
    def _gradient(self, W, b, X, Y):
        m = X.shape[0]
        err = X @ W + b - Y                  # (m, k)
        dW = (X.T @ err) / m + self._penalty_gradient(W)
        db = err.mean(axis=0)                # (k,)
        return dW, db

    def _cost(self, W, b, X, Y) -> float:
        m = X.shape[0]
        err = X @ W + b - Y
        return 0.5 * float(np.mean(err ** 2)) + self._penalty_cost(W)
