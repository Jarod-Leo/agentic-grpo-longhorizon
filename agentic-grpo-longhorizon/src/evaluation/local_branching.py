"""Shared helpers for frozen-policy local branching diagnostics."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentData


def clone_agent_data(source: AgentData, request_id: str) -> AgentData:
    """Copy trajectory-local policy state while retaining shared services."""
    cloned = AgentData(
        messages=copy.deepcopy(source.messages),
        image_data=copy.deepcopy(source.image_data),
        metrics={},
        request_id=request_id,
        tools_kwargs=copy.deepcopy(source.tools_kwargs),
        interaction=source.interaction,
        interaction_kwargs=copy.deepcopy(source.interaction_kwargs),
    )
    for name in (
        "prompt_ids",
        "response_ids",
        "response_mask",
        "response_logprobs",
        "turn_scores",
        "tool_rewards",
        "reasoning_tokens_per_turn",
        "tool_calls",
    ):
        setattr(cloned, name, copy.deepcopy(getattr(source, name)))
    cloned.total_tool_calls = source.total_tool_calls
    cloned.total_errors = source.total_errors
    cloned.user_turns = source.user_turns
    cloned.assistant_turns = source.assistant_turns
    return cloned


def leave_one_out_advantages(candidate_rewards: list[float]) -> list[float]:
    """Return unnormalised sibling advantages for one shared prefix."""
    if len(candidate_rewards) < 2:
        raise ValueError("Local branching requires at least two candidates")
    total = sum(candidate_rewards)
    return [
        reward - (total - reward) / (len(candidate_rewards) - 1)
        for reward in candidate_rewards
    ]


def branch_output_path(output_dir: str | Path) -> Path:
    """Use one append-only file per Ray worker process."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"worker-{os.getpid()}.jsonl"


def append_branch_group(output_dir: str | Path, record: dict[str, Any]) -> Path:
    """Append one complete shared-prefix group without cross-worker writes."""
    path = branch_output_path(output_dir)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
