from __future__ import annotations
from typing import Any

import numpy as np

__all__ = [
    "as_2d_float_array",
    "as_column_vector",
    "check_non_negative",
    "check_positive",
]


def check_positive(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}.")
    return float(value)


def check_non_negative(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return float(value)

def as_2d_float_array(arr: Any, name: str = "X") -> np.ndarray:
    """Coerce arr to a 2-D float64 ndarray, validating shape & finiteness."""
    X = np.asarray(arr, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {X.ndim}-D input.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or +/- inf; clean the data first.")
    return X


def as_column_vector(arr: Any, name: str = "y") -> np.ndarray:
    """Coerce targets to a 2-D (n_samples, n_outputs) float64 array."""
    Y = np.asarray(arr, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    elif Y.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {Y.ndim}-D input.")
    if not np.all(np.isfinite(Y)):
        raise ValueError(f"{name} contains NaN or +/- inf; clean the data first.")
    return Y
