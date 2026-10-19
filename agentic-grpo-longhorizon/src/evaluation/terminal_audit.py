"""Shared audit checks for successful terminal tool calls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.run_metrics import as_timestamp


def summarize_terminal_behavior(
    audit_rows: list[dict[str, Any]], api_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report generation, tool, or user-API work after a successful transfer."""
    first_terminal_by_trajectory: dict[str, float] = {}
    successful_transfers = 0
    terminal_timestamps_valid = True

    for row in audit_rows:
        if not (
            row.get("event") == "execution"
            and row.get("name") == "transfer_to_human_agents"
            and row.get("error") is False
        ):
            continue
        successful_transfers += 1
        trajectory_id = row.get("trajectory_id")
        timestamp = as_timestamp(row.get("time"))
        if not isinstance(trajectory_id, str) or timestamp is None:
            terminal_timestamps_valid = False
            continue
        first_terminal_by_trajectory[trajectory_id] = min(
            timestamp, first_terminal_by_trajectory.get(trajectory_id, timestamp)
        )

    post_terminal_audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        trajectory_id = row.get("trajectory_id")
        terminal_time = first_terminal_by_trajectory.get(trajectory_id)
        timestamp = as_timestamp(row.get("time"))
        if (
            terminal_time is not None
            and timestamp is not None
            and timestamp > terminal_time
            and row.get("event") in {"generation", "execution"}
        ):
            post_terminal_audit[trajectory_id].append(row)

    post_terminal_api: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in api_rows:
        trajectory_id = row.get("trajectory_id")
        terminal_time = first_terminal_by_trajectory.get(trajectory_id)
        started_at = as_timestamp(row.get("started_at"))
        if terminal_time is not None and started_at is not None and started_at > terminal_time:
            post_terminal_api[trajectory_id].append(row)

    post_terminal_generation_events = sum(
        row.get("event") == "generation"
        for rows in post_terminal_audit.values()
        for row in rows
    )
    post_terminal_execution_events = sum(
        row.get("event") == "execution"
        for rows in post_terminal_audit.values()
        for row in rows
    )
    post_terminal_api_requests = sum(map(len, post_terminal_api.values()))
    violating_trajectory_ids = sorted(set(post_terminal_audit) | set(post_terminal_api))

    return {
        "passed": terminal_timestamps_valid and not violating_trajectory_ids,
        "successful_terminal_transfers": successful_transfers,
        "terminal_trajectories": len(first_terminal_by_trajectory),
        "terminal_timestamps_valid": terminal_timestamps_valid,
        "post_terminal_generation_events": post_terminal_generation_events,
        "post_terminal_execution_events": post_terminal_execution_events,
        "post_terminal_api_requests": post_terminal_api_requests,
        "violating_trajectory_ids": violating_trajectory_ids,
        "api_timing_definition": "Only requests started after terminal execution are violations.",
    }
