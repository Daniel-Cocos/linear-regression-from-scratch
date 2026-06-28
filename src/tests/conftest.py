from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def linear_data(rng):
    """A clean, well-conditioned 3-feature regression problem."""
    n, d = 200, 3
    X = rng.normal(0, 1, size=(n, d))
    # Scale features to different magnitudes (internal scaling should fix this).
    X *= np.array([1.0, 10.0, 0.1])
    true_w = np.array([3.0, -2.0, 1.5])
    y = X @ true_w + 4.0 + rng.normal(0, 0.5, size=n)
    return X, y, true_w


@pytest.fixture(scope="session")
def sparse_data(rng):
    """Only the first two features carry signal; the rest are pure noise.

    A correctly-implemented Lasso should drive the noise weights to ~zero.
    """
    n, d = 300, 8
    X = rng.normal(0, 1, size=(n, d))
    true_w = np.array([4.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ true_w + rng.normal(0, 0.5, size=n)
    return X, y, true_w
