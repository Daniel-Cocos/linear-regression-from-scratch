from __future__ import annotations

from .base import BaseRegression
from .models import LassoRegression, LinearRegression, RidgeRegression
from .preprocessing import PolynomialFeatures, StandardScaler
from .solvers import SOLVERS

__version__ = "1.0.0"

__all__ = [
    "BaseRegression",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "StandardScaler",
    "PolynomialFeatures",
    "SOLVERS",
    "__version__",
]
