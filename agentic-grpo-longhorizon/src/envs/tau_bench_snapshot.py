"""Snapshot helpers for branching a live tau-bench interaction.

Only trajectory-local mutable state is copied.  Service clients and static
environment configuration remain shared, so objects containing locks or
connection pools are never deep-copied.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TauBenchInteractionSnapshot:
    """Opaque, reusable snapshot captured at an assistant-turn boundary."""

    env: Any
    state: dict[str, Any]
    initial_user: str


def _copy_user(user: Any, request_id: str | None = None) -> Any:
    """Copy simulator history while retaining its shared API client."""
    cloned = copy.copy(user)
    if hasattr(user, "messages"):
        cloned.messages = copy.deepcopy(user.messages)
    if request_id is not None and hasattr(cloned, "trajectory_id"):
        cloned.trajectory_id = request_id
    return cloned


def _copy_env(env: Any, request_id: str | None = None) -> Any:
    """Copy the mutable tau-bench fields without replaying any action."""
    cloned = copy.copy(env)
    cloned.data = copy.deepcopy(env.data)
    cloned.actions = copy.deepcopy(env.actions)
    cloned.task = copy.deepcopy(env.task)
    cloned.user = _copy_user(env.user, request_id=request_id)
    return cloned


def capture_interaction(entry: dict[str, Any]) -> TauBenchInteractionSnapshot:
    """Capture one live ``TauBenchInteraction._instance_dict`` entry."""
    return TauBenchInteractionSnapshot(
        env=_copy_env(entry["env"]),
        state=copy.deepcopy(entry["state"]),
        initial_user=entry["initial_user"],
    )


def restore_interaction(
    snapshot: TauBenchInteractionSnapshot, request_id: str
) -> dict[str, Any]:
    """Build an independent live entry from ``snapshot``."""
    return {
        "env": _copy_env(snapshot.env, request_id=request_id),
        "state": copy.deepcopy(snapshot.state),
        "initial_user": snapshot.initial_user,
    }
