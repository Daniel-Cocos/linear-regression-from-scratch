from __future__ import annotations

__all__ = [
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
