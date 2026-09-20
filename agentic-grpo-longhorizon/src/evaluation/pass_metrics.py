"""Pure-Python pass metrics for repeated task attempts."""

from __future__ import annotations

from collections.abc import Iterable
from math import comb


def _validate_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def task_pass_metrics(
    n: int,
    c: int,
    ks: Iterable[int] = (1, 4, 8),
) -> dict[str, float | None | dict[int, float | None]]:
    """Compute per-task success, pass@k, and all-success pass^k metrics.

    ``pass_at_k`` is the probability that at least one of ``k`` attempts
    succeeds. ``pass_all_k`` is the probability that all ``k`` attempts
    succeed. Both use sampling-without-replacement estimators over ``n``
    observed attempts and are unavailable when ``n < k``.
    """
    _validate_count("n", n)
    _validate_count("c", c)
    if c > n:
        raise ValueError("c must not exceed n")

    requested_ks = tuple(ks)
    for k in requested_ks:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("each k must be an integer")
        if k <= 0:
            raise ValueError("each k must be positive")

    pass_at_k: dict[int, float | None] = {}
    pass_all_k: dict[int, float | None] = {}
    for k in requested_ks:
        if n < k:
            pass_at_k[k] = None
            pass_all_k[k] = None
            continue

        denominator = comb(n, k)
        pass_at_k[k] = 1.0 - comb(n - c, k) / denominator if n - c >= k else 1.0
        pass_all_k[k] = comb(c, k) / denominator if c >= k else 0.0

    return {
        "success_rate": c / n if n else None,
        "pass_at_k": pass_at_k,
        "pass_all_k": pass_all_k,
    }


def _mean_available(values: Iterable[float | None]) -> float | None:
    """Average available task metrics without treating missing values as zero."""
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None
