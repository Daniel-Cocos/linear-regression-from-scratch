from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["soft_threshold", "SOLVERS"]

SOLVERS: tuple[str, ...] = ("batch", "sgd", "minibatch", "momentum", "adam")

Gradient = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
Cost = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
Prox = Callable[[np.ndarray], np.ndarray]


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding -- the proximal operator of ``alpha * ||.||_1``."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
