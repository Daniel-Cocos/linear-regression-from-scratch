from __future__ import annotations

from .base import BaseRegression
from .preprocessing import StandardScaler
from .solvers import SOLVERS

__version__ = "1.0.0"
__all__ = [
    "BaseRegression",
    "StandardScaler",
    "SOLVERS",
    "__version__",
]
