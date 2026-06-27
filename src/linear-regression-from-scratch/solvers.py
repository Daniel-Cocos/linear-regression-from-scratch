from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["soft_threshold", "SOLVERS", "minimize"]

SOLVERS: tuple[str, ...] = ("batch", "sgd", "minibatch", "momentum", "adam")

Gradient = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]
Cost = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
Prox = Callable[[np.ndarray], np.ndarray]


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def minimize(
    gradient: Gradient,
    X: np.ndarray,
    Y: np.ndarray,
    W0: np.ndarray,
    b0: np.ndarray,
    *,
    solver: str = "batch",
    lr: float = 0.1,
    n_iter: int = 1000,
    tol: float = 1e-8,
    batch_size: int = 32,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    prox_fn: Prox | None = None,
    cost_fn: Cost | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = _as_rng(rng)
    W = np.array(W0, dtype=np.float64, copy=True)
    b = np.array(b0, dtype=np.float64, copy=True)
    m = X.shape[0]
    history: list[float] = []

    # Optimiser state (allocated once; unused rules simply ignore it).
    Vw = np.zeros_like(W)
    Vb = np.zeros_like(b)
    Mw = np.zeros_like(W)
    Mb = np.zeros_like(b)
    Sw = np.zeros_like(W)
    Sb = np.zeros_like(b)

    if solver == "sgd":
        bs = 1
    elif solver == "minibatch":
        bs = max(1, min(int(batch_size), m))
    else:
        bs = m

    prev_cost: float | None = None
    for t in range(1, n_iter + 1):
        # sample a mini-batch when the solver is stochastic
        if bs < m:
            idx = rng.integers(0, m, size=bs)
            Xb, Yb = X[idx], Y[idx]
        else:
            Xb, Yb = X, Y

        dW, db = gradient(W, b, Xb, Yb)

        # parameter update rule
        if solver == "momentum":
            Vw = momentum * Vw - lr * dW
            Vb = momentum * Vb - lr * db
            W += Vw
            b += Vb
        elif solver == "adam":
            Mw = beta1 * Mw + (1 - beta1) * dW
            Mb = beta1 * Mb + (1 - beta1) * db
            Sw = beta2 * Sw + (1 - beta2) * (dW * dW)
            Sb = beta2 * Sb + (1 - beta2) * (db * db)
            mhat = Mw / (1.0 - beta1**t)
            mbhat = Mb / (1.0 - beta1**t)
            vhat = Sw / (1.0 - beta2**t)
            vbhat = Sb / (1.0 - beta2**t)
            W -= lr * mhat / (np.sqrt(vhat) + eps)
            b -= lr * mbhat / (np.sqrt(vbhat) + eps)
        else:  # batch / sgd / minibatch -> vanilla gradient step
            W -= lr * dW
            b -= lr * db

        # proximal step (Lasso)
        if prox_fn is not None:
            W = prox_fn(W)

        # track the full-data objective & early-stop
        if cost_fn is not None:
            c = float(cost_fn(W, b, X, Y))
            history.append(c)
            if prev_cost is not None and abs(prev_cost - c) <= tol * max(
                1.0, abs(prev_cost)
            ):
                break
            prev_cost = c

    return W, b, history
