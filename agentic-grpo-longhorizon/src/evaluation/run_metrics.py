"""Shared task metrics, audit summaries and run-file readers."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.pass_metrics import task_pass_metrics


def read_json(path: Path, default: Any) -> tuple[Any, list[str]]:
    if not path.exists():
        return default, [f"missing:{path.name}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as error:
        return default, [f"invalid:{path.name}:{error}"]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing:{path.name}"]
    rows = []
    errors = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append(f"invalid:{path.name}:{line_number}:not_an_object")
        except json.JSONDecodeError as error:
            errors.append(f"invalid:{path.name}:{line_number}:{error.msg}")
    return rows, errors


def mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.fmean(available) if available else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def as_timestamp(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_epoch(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        value = float(path.read_text(encoding="utf-8").strip())
        return value if math.isfinite(value) else None
    except (OSError, ValueError):
        return None


def is_binary_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value in (0, 1)
    )


def metric_value(mapping: dict[Any, Any], key: int) -> float | None:
    return mapping.get(key, mapping.get(str(key)))


def per_task_results(
    trajectories: list[dict[str, Any]],
    task_splits: dict[int, str],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        task_id = row.get("task_id")
        if (
            task_id in task_splits
            and row.get("split") == task_splits[task_id]
            and is_binary_score(row.get("score"))
        ):
            grouped[task_id].append(row)

    results = []
    for task_id in sorted(task_splits):
        rows = grouped[task_id]
        n = len(rows)
        c = sum(int(row["score"]) for row in rows)
        metrics = task_pass_metrics(n, c, ks=(1, 4, 8))
        results.append(
            {
                "task_id": task_id,
                "split": task_splits[task_id],
                "n": n,
                "c": c,
                "success_rate": metrics["success_rate"],
                "pass_at_k": {
                    str(k): value for k, value in metrics["pass_at_k"].items()
                },
                "pass_all_k": {
                    str(k): value for k, value in metrics["pass_all_k"].items()
                },
            }
        )
    return results


def aggregate_split(task_rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    rows = [row for row in task_rows if row["split"] == split_name]
    return {
        "tasks": len(rows),
        "n": sum(row["n"] for row in rows),
        "c": sum(row["c"] for row in rows),
        "success_rate": mean([row["success_rate"] for row in rows]),
        "pass_at_4": mean([metric_value(row["pass_at_k"], 4) for row in rows]),
        "pass_all_4": mean([metric_value(row["pass_all_k"], 4) for row in rows]),
    }


def summarize_tools(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    generations = [row for row in audit_rows if row.get("event") == "generation"]
    executions = [row for row in audit_rows if row.get("event") == "execution"]
    attempts = sum(row.get("tool_tag_starts", 0) for row in generations)
    strict_valid = sum(
        call.get("json_valid") is True
        and call.get("known_name") is True
        and call.get("schema_valid") is True
        for row in generations
        for call in row.get("calls", [])
    )
    return {
        "generation_events": len(generations),
        "attempts": attempts,
        "executions": len(executions),
        "strict_valid": strict_valid,
        "strict_valid_rate": strict_valid / attempts if attempts else None,
        "error_responses": sum(row.get("error") is True for row in executions),
        "nonempty_thinking_events": sum(
            row.get("nonempty_thinking") is True for row in generations
        ),
    }


def summarize_api(api_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (row for row in api_rows if as_timestamp(row.get("started_at")) is not None),
        key=lambda row: as_timestamp(row["started_at"]),
    )
    usage_names = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {
        name: sum((row.get("usage") or {}).get(name, 0) or 0 for row in api_rows)
        for name in usage_names
    }
    latencies = [
        float(row["latency_s"])
        for row in api_rows
        if isinstance(row.get("latency_s"), (int, float))
    ]
    window: deque[dict[str, Any]] = deque()
    peak_rpm = 0
    peak_tpm = 0
    for row in ordered:
        started = as_timestamp(row["started_at"])
        window.append(row)
        while window and as_timestamp(window[0]["started_at"]) <= started - 60:
            window.popleft()
        peak_rpm = max(peak_rpm, len(window))
        peak_tpm = max(
            peak_tpm,
            sum(
                (item.get("usage") or {}).get("total_tokens", 0) or 0 for item in window
            ),
        )
    if ordered:
        first = as_timestamp(ordered[0]["started_at"])
        finishes = [as_timestamp(row.get("finished_at")) for row in ordered]
        valid_finishes = [value for value in finishes if value is not None]
        span = max(valid_finishes) - first if valid_finishes else None
    else:
        span = None
    return {
        "requests": len(api_rows),
        "successful_requests": sum(row.get("success") is True for row in api_rows),
        "failed_attempts": sum(row.get("success") is False for row in api_rows),
        "retry_attempts": sum((row.get("attempt") or 1) > 1 for row in api_rows),
        "usage": usage,
        "latency_s": {
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "p99": percentile(latencies, 0.99),
        },
        "active_span_s": span,
        "mean_rpm": len(api_rows) * 60 / span if span and span > 0 else None,
        "peak_rolling_60s_rpm": peak_rpm,
        "peak_rolling_60s_tpm": peak_tpm,
    }


def contains_training_metric(row: dict[str, Any]) -> bool:
    data = row.get("data", row)
    if not isinstance(data, dict):
        return False
    return any(str(key).startswith("actor/") or "grad_norm" in str(key) for key in data)


def checkpoint_files(run: Path) -> list[str]:
    found = []
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if any("checkpoint" in part.lower() for part in relative.parts):
            found.append(str(relative))
    return sorted(found)
