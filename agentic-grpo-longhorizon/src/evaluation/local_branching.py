"""Shared helpers for frozen-policy local branching diagnostics."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verl.experimental.agent_loop.tool_agent_loop import AgentData


def clone_agent_data(source: AgentData, request_id: str) -> AgentData:
    """Copy trajectory-local policy state while retaining shared services."""
    from verl.experimental.agent_loop.tool_agent_loop import AgentData

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


def read_training_branch_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read all per-worker branch records from a training or signal run."""
    records = []
    for path in sorted((Path(run_dir) / "branches").glob("worker-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    return records


def summarize_training_branches(
    records: list[dict[str, Any]],
    root_rows: list[dict[str, Any]],
    *,
    candidates: int,
    continuations: int,
) -> dict[str, Any]:
    """Validate C branch records and expose audit IDs for auxiliary rollouts."""
    root_by_id = {
        row.get("trajectory_id"): row
        for row in root_rows
        if isinstance(row.get("trajectory_id"), str)
    }
    checks = {
        "schema": True,
        "root_link": True,
        "root_rollout_only": True,
        "candidate0_matches_root_action": True,
        "branch_shape": True,
        "binary_outcomes": True,
        "derived_q_and_advantage": True,
        "cost_totals": True,
    }
    allowed_ids: set[str] = set()
    complete = []
    skipped = []
    split_half_stable_groups = 0
    pooled_root_advantages: list[float | None] = []
    trigger_counts: dict[str, int] = {}
    total_cost = {
        "sampled_sibling_actions": 0,
        "sibling_candidate_policy_tokens": 0,
        "suffix_policy_tokens": 0,
        "branch_user_turns": 0,
        "branch_tool_calls": 0,
        "branch_episodes": 0,
        "wall_time_s": 0.0,
    }

    for record in records:
        checks["schema"] &= (
            record.get("schema_version") == "local_branch_root_action_v1"
            and record.get("record_type") == "group"
        )
        root = root_by_id.get(record.get("root_trajectory_id"))
        checks["root_link"] &= (
            root is not None
            and root.get("task_id") == record.get("task_id")
            and root.get("split") == record.get("split")
        )
        checks["root_rollout_only"] &= record.get("rollout_n") == 0
        if record.get("skipped") is True:
            skipped.append(record)
            continue

        complete.append(record)
        trigger = record.get("trigger")
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        rows = record.get("candidates", [])
        shape = (
            len(rows) == candidates
            and [row.get("candidate_index") for row in rows]
            == list(range(candidates))
            and all(
                len(row.get("continuations", [])) == continuations
                for row in rows
            )
        )
        checks["branch_shape"] &= shape
        if not shape:
            continue
        root_tokens = record.get("root_action_token_ids")
        root_span = record.get("root_action_span")
        candidate0 = rows[0]
        checks["candidate0_matches_root_action"] &= (
            candidate0.get("is_root_action") is True
            and candidate0.get("candidate_token_ids") == root_tokens
            and isinstance(root_span, list)
            and len(root_span) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in root_span
            )
            and isinstance(root_tokens, list)
            and bool(root_tokens)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in root_tokens
            )
            and root_span[1] - root_span[0] == len(root_tokens or [])
        )
        means = []
        rewards_by_candidate = []
        suffix_tokens = 0
        for index, candidate in enumerate(rows):
            candidate_id = candidate.get("candidate_request_id")
            expected_candidate_id = (
                f"{record['root_trajectory_id']}-branch-c{index}"
            )
            checks["branch_shape"] &= candidate.get("is_root_action") is (
                index == 0
            )
            checks["branch_shape"] &= (
                candidate_id == expected_candidate_id
                and isinstance(candidate.get("candidate_token_ids"), list)
                and bool(candidate["candidate_token_ids"])
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in candidate["candidate_token_ids"]
                )
                and candidate.get("candidate_policy_tokens")
                == len(candidate["candidate_token_ids"])
            )
            rewards = [
                continuation.get("outcome_reward")
                for continuation in candidate["continuations"]
            ]
            binary = all(value in (0, 1, 0.0, 1.0) for value in rewards)
            checks["binary_outcomes"] &= binary
            mean = sum(float(value) for value in rewards) / continuations
            means.append(mean)
            rewards_by_candidate.append([float(value) for value in rewards])
            checks["derived_q_and_advantage"] &= math.isclose(
                float(candidate.get("mean_reward", math.nan)), mean
            )
            suffix_tokens += sum(
                int(item.get("suffix_policy_tokens", 0))
                for item in candidate["continuations"]
            )
            allowed_ids.add(candidate_id)
            for repeat_index, continuation in enumerate(
                candidate["continuations"]
            ):
                expected_branch_id = (
                    f"{record['root_trajectory_id']}-branch-c{index}"
                    f"-m{repeat_index}"
                )
                checks["branch_shape"] &= (
                    continuation.get("repeat_index") == repeat_index
                    and continuation.get("trajectory_id")
                    == expected_branch_id
                )
                allowed_ids.add(continuation.get("trajectory_id"))
        root_advantage = means[0] - sum(means[1:]) / (candidates - 1)
        checks["derived_q_and_advantage"] &= math.isclose(
            float(record.get("root_local_advantage", math.nan)),
            root_advantage,
        )
        checks["derived_q_and_advantage"] &= math.isclose(
            float(candidate0.get("local_advantage", math.nan)),
            root_advantage,
        )
        root_action = tuple(root_tokens or [])

        def pooled_root_advantage(
            repeat_slice: slice | None = None,
        ) -> float | None:
            rewards_by_action: dict[tuple[int, ...], list[float]] = {}
            for candidate, rewards in zip(
                rows, rewards_by_candidate, strict=True
            ):
                action = tuple(candidate.get("candidate_token_ids", []))
                selected = (
                    rewards
                    if repeat_slice is None
                    else rewards[repeat_slice]
                )
                rewards_by_action.setdefault(action, []).extend(selected)
            if len(rewards_by_action) < 2 or root_action not in rewards_by_action:
                return None
            action_means = {
                action: sum(values) / len(values)
                for action, values in rewards_by_action.items()
            }
            siblings = [
                value
                for action, value in action_means.items()
                if action != root_action
            ]
            return action_means[root_action] - sum(siblings) / len(siblings)

        pooled_advantage = pooled_root_advantage()
        pooled_root_advantages.append(pooled_advantage)
        if continuations >= 4:
            midpoint = continuations // 2
            half_advantages = [
                pooled_root_advantage(slice(0, midpoint)),
                pooled_root_advantage(slice(midpoint, continuations)),
            ]
            if (
                half_advantages[0] is not None
                and half_advantages[1] is not None
                and not math.isclose(half_advantages[0], 0.0)
                and not math.isclose(half_advantages[1], 0.0)
                and half_advantages[0] * half_advantages[1] > 0
            ):
                split_half_stable_groups += 1

        cost = record.get("cost", {})
        expected_sibling_tokens = sum(
            int(item.get("candidate_policy_tokens", 0)) for item in rows[1:]
        )
        checks["cost_totals"] &= (
            cost.get("root_action_tokens") == len(root_tokens or [])
            and cost.get("sampled_sibling_actions") == candidates - 1
            and cost.get("sibling_candidate_policy_tokens")
            == expected_sibling_tokens
            and cost.get("suffix_policy_tokens") == suffix_tokens
            and cost.get("branch_episodes") == candidates * continuations
        )
        for name in total_cost:
            total_cost[name] += cost.get(name, 0)

    advantages = [float(row["root_local_advantage"]) for row in complete]
    raw_nonzero = [
        value for value in advantages if not math.isclose(value, 0.0)
    ]
    nonzero = [
        value
        for value in pooled_root_advantages
        if value is not None and not math.isclose(value, 0.0)
    ]
    allowed_ids.discard(None)
    return {
        "schema_version": "local_branch_training_summary_v1",
        "records": len(records),
        "complete_groups": len(complete),
        "skipped_groups": len(skipped),
        "raw_nonzero_advantage_groups": len(raw_nonzero),
        "nonzero_signal_groups": len(nonzero),
        "zero_signal_groups": len(complete) - len(nonzero),
        "split_half_stable_signal_groups": split_half_stable_groups,
        "nonzero_signal_rate": len(nonzero) / len(complete) if complete else None,
        "mean_abs_root_advantage": (
            sum(abs(value) for value in advantages) / len(advantages)
            if advantages
            else None
        ),
        "mean_abs_pooled_root_advantage": (
            sum(abs(value) for value in nonzero) / len(nonzero)
            if nonzero
            else None
        ),
        "trigger_counts": trigger_counts,
        "cost": total_cost,
        "checks": checks,
        "engineering_accepted": bool(records) and all(checks.values()),
        "allowed_auxiliary_trajectory_ids": sorted(allowed_ids),
    }
