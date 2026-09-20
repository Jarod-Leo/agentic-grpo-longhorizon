from __future__ import annotations

import pytest
from src.evaluation.pass_metrics import _mean_available, task_pass_metrics


@pytest.mark.parametrize(
    ("successes", "expected_at", "expected_all"),
    [
        (1, {1: 0.125, 4: 0.5, 8: 1.0}, {1: 0.125, 4: 0.0, 8: 0.0}),
        (7, {1: 0.875, 4: 1.0, 8: 1.0}, {1: 0.875, 4: 0.5, 8: 0.0}),
        (0, {1: 0.0, 4: 0.0, 8: 0.0}, {1: 0.0, 4: 0.0, 8: 0.0}),
        (8, {1: 1.0, 4: 1.0, 8: 1.0}, {1: 1.0, 4: 1.0, 8: 1.0}),
    ],
)
def test_task_pass_metrics_for_eight_attempts(successes, expected_at, expected_all):
    metrics = task_pass_metrics(8, successes)

    assert metrics["success_rate"] == successes / 8
    assert metrics["pass_at_k"] == expected_at
    assert metrics["pass_all_k"] == expected_all


def test_metrics_are_missing_when_task_has_fewer_than_k_attempts():
    metrics = task_pass_metrics(2, 1)

    assert metrics["pass_at_k"] == {1: 0.5, 4: None, 8: None}
    assert metrics["pass_all_k"] == {1: 0.5, 4: None, 8: None}


def test_macro_mean_ignores_missing_task_metrics():
    assert _mean_available([None, 0.25, None, 0.75]) == 0.5
    assert _mean_available([None, None]) is None


@pytest.mark.parametrize(
    ("n", "c", "error"),
    [
        (-1, 0, ValueError),
        (2, -1, ValueError),
        (2, 3, ValueError),
        (1.0, 0, TypeError),
        (2, True, TypeError),
    ],
)
def test_invalid_counts_are_rejected(n, c, error):
    with pytest.raises(error):
        task_pass_metrics(n, c)


def test_zero_attempts_have_no_success_rate():
    metrics = task_pass_metrics(0, 0)

    assert metrics["success_rate"] is None
    assert all(value is None for value in metrics["pass_at_k"].values())
    assert all(value is None for value in metrics["pass_all_k"].values())
