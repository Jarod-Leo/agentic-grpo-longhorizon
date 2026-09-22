"""Grounded, privacy-preserving feedback for Agent-OPSD self-distillation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from numbers import Real
from typing import Any

OPSD_FEEDBACK_VERSION = "opsd-f2-v1"
OPSD_FEEDBACK_MODE = "F2"

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TERMINATION_TEXT = {
    "response_budget": "the response token budget was reached",
    "assistant_turn_limit": "the assistant turn limit was reached",
    "user_turn_limit": "the user turn limit was reached",
    "tool_response_budget": "the tool response token budget was reached",
    "interaction_terminated": "the environment interaction terminated",
}


class OpsdFeedbackError(ValueError):
    """Raised when trajectory state cannot support grounded OPSD feedback."""


class ContaminatedTrajectoryError(OpsdFeedbackError):
    """Raised when a contaminated trajectory must be rejected from training."""


def empty_opsd_feedback(version: str = OPSD_FEEDBACK_VERSION) -> dict[str, Any]:
    """Return the explicit F0 negative-control payload."""
    if version != OPSD_FEEDBACK_VERSION:
        raise OpsdFeedbackError(f"unsupported feedback version: {version!r}")
    return {
        "version": OPSD_FEEDBACK_VERSION,
        "mode": "F0",
        "text": "",
        "rules": [],
    }


def _binary_outcome(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or float(value) not in (0.0, 1.0)
    ):
        raise OpsdFeedbackError("outcome_reward must be a binary numeric scalar")
    return int(value)


def _tool_name(action: Mapping[str, Any]) -> str:
    name = action.get("tool")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise OpsdFeedbackError("action_history contains an invalid tool name")
    return name


def _parameter_signature(action: Mapping[str, Any]) -> str:
    parameters = action.get("parameters")
    if not isinstance(parameters, Mapping):
        raise OpsdFeedbackError("failed tool action is missing a parameter mapping")
    try:
        return json.dumps(
            parameters, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise OpsdFeedbackError(
            "failed tool parameters are not JSON serializable"
        ) from error


def build_opsd_feedback(
    state: Mapping[str, Any],
    *,
    outcome_reward: Any,
    termination: str,
    version: str = OPSD_FEEDBACK_VERSION,
) -> dict[str, Any]:
    """Build F2 feedback only from auditable terminal and tool-call state.

    Parameters are used solely to group exactly repeated failed calls. They
    are never rendered into the returned feedback.
    """

    if not isinstance(state, Mapping):
        raise OpsdFeedbackError("trajectory state must be a mapping")
    if state.get("contaminated") is not False:
        raise ContaminatedTrajectoryError(
            "contaminated trajectory cannot produce OPSD feedback"
        )
    if version != OPSD_FEEDBACK_VERSION:
        raise OpsdFeedbackError(f"unsupported feedback version: {version!r}")
    if termination not in _TERMINATION_TEXT:
        raise OpsdFeedbackError(f"unsupported terminal state: {termination!r}")

    outcome = _binary_outcome(outcome_reward)
    history = state.get("action_history")
    if not isinstance(history, list):
        raise OpsdFeedbackError("trajectory state is missing action_history")

    failed_calls: list[tuple[str, str]] = []
    failed_tools: list[str] = []
    for action in history:
        if not isinstance(action, Mapping):
            raise OpsdFeedbackError("action_history entries must be mappings")
        is_error = action.get("is_error")
        if not isinstance(is_error, bool):
            raise OpsdFeedbackError("action_history is_error must be a boolean")
        if not is_error:
            continue
        name = _tool_name(action)
        failed_tools.append(name)
        failed_calls.append((name, _parameter_signature(action)))

    lines = [
        f"Terminal outcome: {'success' if outcome else 'failure'}.",
        f"Termination: {_TERMINATION_TEXT[termination]}.",
    ]
    rules = ["terminal_outcome", "termination_reason"]

    for name in dict.fromkeys(failed_tools):
        lines.append(
            f"Tool error observed for {name}. Review the returned error and change the next call when needed."
        )
    if failed_tools:
        rules.append("tool_error")

    repeated = Counter(failed_calls)
    for (name, _signature), count in repeated.items():
        if count >= 2:
            lines.append(
                f"The same failed {name} call was repeated {count} times. Change the call before retrying."
            )
    if any(count >= 2 for count in repeated.values()):
        rules.append("repeated_failed_call")

    return {
        "version": OPSD_FEEDBACK_VERSION,
        "mode": OPSD_FEEDBACK_MODE,
        "text": "\n".join(lines),
        "rules": rules,
    }
