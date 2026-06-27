from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .preprocessing import StandardScaler
from .solvers import SOLVERS, minimize
from .utils import (
    as_2d_float_array,
    as_column_vector,
    check_fitted,
    check_non_negative,
    check_positive,
)

__all__ = ["BaseRegression"]

log = logging.getLogger("linear_regression_from_scratch")
_ALL_SOLVERS = (*SOLVERS, "normal")


class BaseRegression:
    """Shared linear regression engine. Use a concrete subclass instead"""

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

        # Learned attributes (populated by fit)
        self.w_: np.ndarray | None = None
        self.b_: float | np.ndarray | None = None
        self.cost_history_: list[float] = []
        self.n_iter_: int = 0
        self.scaler_: StandardScaler | None = None
        self.n_features_in_: int | None = None

    # Penalty hooks overridden by subclasses
    def _penalty_gradient(self, W: np.ndarray) -> np.ndarray:
        return np.zeros_like(W)

    def _penalty_cost(self, W: np.ndarray) -> float:
        return 0.0

    def _prox(self, W: np.ndarray) -> np.ndarray:
        return W

    # Gradients & cost on the (smooth) data term
    def _gradient(self, W, b, X, Y):
        m = X.shape[0]
        err = X @ W + b - Y  # (m, k)
        dW = (X.T @ err) / m + self._penalty_gradient(W)
        db = err.mean(axis=0)  # (k,)
        return dW, db

    def _cost(self, W, b, X, Y) -> float:
        m = X.shape[0]
        err = X @ W + b - Y
        return 0.5 * float(np.mean(err**2)) + self._penalty_cost(W)

    # Fitting
    def fit(self, X: Any, y: Any) -> "BaseRegression":
        Xr = as_2d_float_array(X, "X")
        Yr = as_column_vector(y, "y")
        if Xr.shape[0] != Yr.shape[0]:
            raise ValueError(f"X has {Xr.shape[0]} samples but y has {Yr.shape[0]}.")
        self.n_features_in_ = Xr.shape[1]
        n_features, n_outputs = Xr.shape[1], Yr.shape[1]

        if self.solver == "normal":
            self._fit_normal(Xr, Yr)
            return self

        # Train on standardised features for numerically stable descent
        if self.standardize:
            self.scaler_ = StandardScaler().fit(Xr)
            Xf = self.scaler_.transform(Xr)
        else:
            self.scaler_ = None
            Xf = Xr

        W0 = np.zeros((n_features, n_outputs))
        b0 = np.zeros(n_outputs)

        W, b, history = minimize(
            self._gradient,
            Xf,
            Yr,
            W0,
            b0,
            solver=self.solver,
            lr=self.lr,
            n_iter=self.n_iter,
            tol=self.tol,
            batch_size=self.batch_size,
            momentum=self.momentum,
            beta1=self.beta1,
            beta2=self.beta2,
            prox_fn=self._prox,
            cost_fn=self._cost,
            rng=self.random_state,
        )
        self.cost_history_ = history
        self.n_iter_ = len(history)

        # Report coefficients in the original feature space
        if self.scaler_ is not None:
            W, b = self.scaler_.unstandardize_weights(W, b)

        self.w_, self.b_ = self._store_params(W, b, n_outputs)

        if self.verbose:
            final = history[-1] if history else float("nan")
            log.info(
                "%s trained in %d iterations | final objective=%.6g",
                type(self).__name__,
                self.n_iter_,
                final,
            )
        return self

    def _fit_normal(self, X: np.ndarray, Y: np.ndarray) -> "BaseRegression":
        """Closed form ordinary least squares via the pseudo inverse"""
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(Xb, Y, rcond=None)  # (n_features + 1, k)
        W, b = coef[:-1], coef[-1]
        self.w_, self.b_ = self._store_params(W, b, Y.shape[1])
        self.cost_history_ = []
        self.n_iter_ = 0
        self.scaler_ = None
        return self

    @staticmethod
    def _store_params(W: np.ndarray, b: np.ndarray, n_outputs: int):
        """Flatten single output params for an intuitive public API"""
        if n_outputs == 1:
            return W[:, 0].copy(), float(b[0])
        return W.copy(), b.copy()

    # Inference & metrics
    def predict(self, X: Any) -> np.ndarray:
        """Predict targets. Returns shape (m,) for single output, (m, k) for multi"""
        check_fitted(self)
        X = as_2d_float_array(X, "X")
        return X @ self.w_ + self.b_  # 1-D w_ -> (m,); 2-D w_ -> (m, k)

    def score(self, X: Any, y: Any) -> float:
        """R^2 (coefficient of determination)."""
        Y = as_column_vector(y, "y")
        Yhat = as_column_vector(self.predict(X))
        ss_res = float(np.sum((Y - Yhat) ** 2))
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    def rmse(self, X: Any, y: Any) -> float:
        Y = as_column_vector(y, "y")
        Yhat = as_column_vector(self.predict(X))
        return float(np.sqrt(np.mean((Y - Yhat) ** 2)))

    def mae(self, X: Any, y: Any) -> float:
        Y = as_column_vector(y, "y")
        Yhat = as_column_vector(self.predict(X))
        return float(np.mean(np.abs(Y - Yhat)))

    # Persistence
    def save(self, path: str) -> None:
        """Persist learned parameters + config to a .npz file"""
        check_fitted(self)
        np.savez(
            path,
            class_=type(self).__name__,
            w_=self.w_,
            b_=self.b_,
            n_features_in_=self.n_features_in_,
            n_iter_=self.n_iter_,
        )

    @classmethod
    def load(cls, path: str) -> "BaseRegression":
        data = np.load(path, allow_pickle=False)
        obj = cls()
        obj.w_ = np.asarray(data["w_"])
        obj.b_ = (
            np.asarray(data["b_"]).item()
            if data["b_"].ndim == 0
            else np.asarray(data["b_"])
        )
        obj.n_features_in_ = int(data["n_features_in_"])
        obj.n_iter_ = int(data["n_iter_"])
        return obj

    def __repr__(self) -> str:
        reg = getattr(self, "alpha", None)
        reg_str = f", alpha={reg}" if reg is not None else ""
        fitted = "fitted" if self.w_ is not None else "unfitted"
        return (
            f"{type(self).__name__}(lr={self.lr}, n_iter={self.n_iter}, "
            f"solver={self.solver!r}{reg_str}) <{fitted}>"
        )
