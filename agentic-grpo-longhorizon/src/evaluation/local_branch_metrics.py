"""Validation and signal summaries for local-branch diagnostic runs."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.run_metrics import (
    is_binary_score,
    read_epoch,
    read_json,
    read_jsonl,
    summarize_api,
    summarize_tools,
)
from src.evaluation.terminal_audit import summarize_terminal_behavior


SCHEMA_VERSION = "local_branch_diagnostic_v1"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _read_branch_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    paths = sorted((run_dir / "branches").glob("worker-*.jsonl"))
    if not paths:
        return [], ["missing:branches/worker-*.jsonl"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        part, part_errors = read_jsonl(path)
        rows.extend(part)
        errors.extend(f"{path.name}:{error}" for error in part_errors)
    return rows, errors


def _candidate_values(candidate: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    continuations = candidate.get("continuations")
    if not isinstance(continuations, list):
        return [], []
    rewards = [
        float(row["outcome_reward"])
        for row in continuations
        if isinstance(row, dict) and is_binary_score(row.get("outcome_reward"))
    ]
    return rewards, [row for row in continuations if isinstance(row, dict)]


def _pooled_action_means(
    candidates: list[dict[str, Any]], repeat_slice: slice | None = None
) -> dict[tuple[int, ...], float]:
    rewards_by_action: dict[tuple[int, ...], list[float]] = {}
    for candidate in candidates:
        action = tuple(candidate.get("candidate_token_ids", []))
        rewards, _ = _candidate_values(candidate)
        selected = rewards if repeat_slice is None else rewards[repeat_slice]
        rewards_by_action.setdefault(action, []).extend(selected)
    return {
        action: sum(rewards) / len(rewards)
        for action, rewards in rewards_by_action.items()
        if rewards
    }


def summarize_action_variation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate raw suffix noise from differences between exact token actions."""
    raw_means = []
    for candidate in candidates:
        rewards, _ = _candidate_values(candidate)
        raw_means.append(sum(rewards) / len(rewards))

    action_means = _pooled_action_means(candidates)
    pooled_candidate_means = [
        action_means[tuple(candidate.get("candidate_token_ids", []))]
        for candidate in candidates
    ]
    return {
        "raw_candidate_mean_rewards": raw_means,
        "pooled_candidate_mean_rewards": pooled_candidate_means,
        "unique_actions": len(action_means),
        "raw_effective": max(raw_means) > min(raw_means),
        "effective": len(action_means) > 1 and max(action_means.values()) > min(action_means.values()),
    }


def summarize_split_half_stability(
    groups: list[dict[str, Any]], continuations_per_candidate: int
) -> dict[str, Any]:
    """Describe whether independently repeated halves identify the same best action."""
    available = continuations_per_candidate >= 4
    midpoint = continuations_per_candidate // 2
    multi_action_prefixes = 0
    informative_in_both_halves = 0
    unique_best_in_both_halves = 0
    same_unique_best_action = 0
    ties_in_either_half = 0

    if available:
        for group in groups:
            candidates = group["candidates"]
            first = _pooled_action_means(candidates, slice(0, midpoint))
            second = _pooled_action_means(candidates, slice(midpoint, continuations_per_candidate))
            if len(first) < 2:
                continue
            multi_action_prefixes += 1
            first_informative = max(first.values()) > min(first.values())
            second_informative = max(second.values()) > min(second.values())
            if not (first_informative and second_informative):
                continue
            informative_in_both_halves += 1
            first_best = {action for action, value in first.items() if value == max(first.values())}
            second_best = {action for action, value in second.items() if value == max(second.values())}
            if len(first_best) != 1 or len(second_best) != 1:
                ties_in_either_half += 1
                continue
            unique_best_in_both_halves += 1
            same_unique_best_action += first_best == second_best

    return {
        "available": available,
        "continuations_per_candidate": continuations_per_candidate,
        "first_half_repeats": midpoint if available else None,
        "second_half_repeats": continuations_per_candidate - midpoint if available else None,
        "multi_action_prefixes": multi_action_prefixes,
        "informative_in_both_halves": informative_in_both_halves,
        "unique_best_in_both_halves": unique_best_in_both_halves,
        "ties_in_either_half": ties_in_either_half,
        "same_unique_best_action": same_unique_best_action,
        "same_unique_best_action_rate": (
            same_unique_best_action / unique_best_in_both_halves
            if unique_best_in_both_halves
            else None
        ),
        "note": "Descriptive split-half stability; it is not a statistical significance test.",
    }


def _validate_complete_group(
    row: dict[str, Any], candidates_per_prefix: int, continuations_per_candidate: int
) -> dict[str, bool]:
    root_id = row.get("root_trajectory_id")
    candidates = row.get("candidates")
    structure = (
        isinstance(candidates, list)
        and len(candidates) == candidates_per_prefix
        and [candidate.get("candidate_index") for candidate in candidates]
        == list(range(candidates_per_prefix))
    )
    if not structure:
        return {
            "branch_shape": False,
            "binary_rewards": False,
            "candidate_tokens_fixed": False,
            "derived_values": False,
            "cost_totals": False,
        }

    binary_rewards = True
    candidate_tokens_fixed = True
    derived_values = True
    suffix_policy_tokens = 0
    branch_episodes = 0
    candidate_policy_tokens = 0
    seen_ids: set[str] = set()
    means = []

    for candidate_index, candidate in enumerate(candidates):
        token_ids = candidate.get("candidate_token_ids")
        candidate_request_id = candidate.get("candidate_request_id")
        policy_tokens = candidate.get("candidate_policy_tokens")
        expected_candidate_id = f"{root_id}-branch-c{candidate_index}"
        candidate_tokens_fixed &= (
            isinstance(token_ids, list)
            and bool(token_ids)
            and all(_is_int(token) for token in token_ids)
            and _is_int(policy_tokens)
            and policy_tokens == len(token_ids)
            and candidate_request_id == expected_candidate_id
        )
        if _is_int(policy_tokens):
            candidate_policy_tokens += policy_tokens

        rewards, continuations = _candidate_values(candidate)
        shape_ok = (
            len(continuations) == continuations_per_candidate
            and [continuation.get("repeat_index") for continuation in continuations]
            == list(range(continuations_per_candidate))
        )
        structure &= shape_ok
        binary_rewards &= len(rewards) == continuations_per_candidate
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        means.append(mean_reward)
        derived_values &= (
            mean_reward is not None
            and _is_number(candidate.get("mean_reward"))
            and math.isclose(float(candidate["mean_reward"]), mean_reward)
        )

        for repeat_index, continuation in enumerate(continuations):
            branch_id = continuation.get("trajectory_id")
            expected_id = f"{root_id}-branch-c{candidate_index}-m{repeat_index}"
            structure &= isinstance(branch_id, str) and branch_id == expected_id
            structure &= branch_id not in seen_ids
            if isinstance(branch_id, str):
                seen_ids.add(branch_id)
            for field in ("suffix_policy_tokens", "suffix_total_tokens"):
                structure &= _is_int(continuation.get(field)) and continuation[field] >= 0
            structure &= _is_number(continuation.get("wall_time_s")) and continuation["wall_time_s"] >= 0
            if _is_int(continuation.get("suffix_policy_tokens")):
                suffix_policy_tokens += continuation["suffix_policy_tokens"]
            branch_episodes += 1

    if all(mean is not None for mean in means):
        for index, candidate in enumerate(candidates):
            other_mean = sum(means[:index] + means[index + 1 :]) / (len(means) - 1)
            derived_values &= _is_number(candidate.get("local_advantage")) and math.isclose(
                float(candidate["local_advantage"]), means[index] - other_mean
            )
        derived_values &= row.get("effective") is (max(means) > min(means))
    else:
        derived_values = False

    cost = row.get("cost")
    cost_totals = isinstance(cost, dict) and (
        cost.get("candidate_policy_tokens") == candidate_policy_tokens
        and cost.get("suffix_policy_tokens") == suffix_policy_tokens
        and cost.get("branch_episodes") == branch_episodes
        and _is_number(cost.get("wall_time_s"))
        and cost["wall_time_s"] >= 0
    )
    return {
        "branch_shape": bool(structure),
        "binary_rewards": bool(binary_rewards),
        "candidate_tokens_fixed": bool(candidate_tokens_fixed),
        "derived_values": bool(derived_values),
        "cost_totals": bool(cost_totals),
    }


def build_local_branch_summary(run_dir: Path) -> dict[str, Any]:
    """Validate one diagnostic run and summarize its local branch signal."""
    run_meta, errors = read_json(run_dir / "run.json", {})
    branch_meta, new_errors = read_json(run_dir / "branch_run.json", {})
    errors.extend(new_errors)
    roots, new_errors = read_jsonl(run_dir / "eval_trajectories.jsonl")
    errors.extend(new_errors)
    api_rows, new_errors = read_jsonl(run_dir / "api.jsonl")
    errors.extend(new_errors)
    audit_rows, new_errors = read_jsonl(run_dir / "tool_audit.jsonl")
    errors.extend(new_errors)
    groups, new_errors = _read_branch_rows(run_dir)
    errors.extend(new_errors)

    expected_backbones = branch_meta.get("expected_backbones")
    candidates_per_prefix = branch_meta.get("candidates")
    continuations_per_candidate = branch_meta.get("continuations")
    anchor_turns = branch_meta.get("anchor_turns")
    target_tasks = branch_meta.get("task_ids")
    train_tasks = run_meta.get("train_task_ids")
    samples_per_task = run_meta.get("samples_per_task")

    metadata_valid = (
        branch_meta.get("stage") in {"smoke", "signal"}
        and _is_int(expected_backbones)
        and expected_backbones > 0
        and _is_int(candidates_per_prefix)
        and candidates_per_prefix >= 2
        and _is_int(continuations_per_candidate)
        and continuations_per_candidate >= 1
        and isinstance(anchor_turns, list)
        and bool(anchor_turns)
        and all(_is_int(turn) and turn >= 1 for turn in anchor_turns)
        and len(set(anchor_turns)) == len(anchor_turns)
        and isinstance(target_tasks, list)
        and bool(target_tasks)
        and all(_is_int(task_id) for task_id in target_tasks)
        and len(set(target_tasks)) == len(target_tasks)
        and isinstance(train_tasks, list)
        and bool(train_tasks)
        and all(_is_int(task_id) for task_id in train_tasks)
        and set(target_tasks).issubset(train_tasks)
        and _is_int(samples_per_task)
        and samples_per_task > 0
    )

    root_ids = [row.get("trajectory_id") for row in roots]
    root_task_counts = Counter(row.get("task_id") for row in roots)
    roots_complete = metadata_valid and (
        len(roots) == expected_backbones == run_meta.get("expected_trajectories")
        and len(root_ids) == len(set(root_ids))
        and None not in root_ids
        and set(root_task_counts) == set(train_tasks)
        and all(root_task_counts[task_id] == samples_per_task for task_id in train_tasks)
        and all(
            row.get("split") == "train"
            and is_binary_score(row.get("score"))
            and row.get("schema_verified") is True
            for row in roots
        )
    )
    root_by_id = {
        row.get("trajectory_id"): row for row in roots if isinstance(row.get("trajectory_id"), str)
    }

    group_root_ids = [row.get("root_trajectory_id") for row in groups]
    group_rollouts_by_task: dict[int, set[int]] = {}
    for row in groups:
        task_id = row.get("task_id")
        rollout_n = row.get("rollout_n")
        if _is_int(task_id) and _is_int(rollout_n):
            group_rollouts_by_task.setdefault(task_id, set()).add(rollout_n)
    group_coverage = (
        metadata_valid
        and bool(groups)
        and len(group_root_ids) == len(set(group_root_ids))
        and set(group_root_ids) == set(root_ids)
        and all(
            _is_int(row.get("rollout_n"))
            and 0 <= row["rollout_n"] < samples_per_task
            and row.get("requested_anchor_turn")
            == anchor_turns[row["rollout_n"] % len(anchor_turns)]
            for row in groups
        )
        and all(
            group_rollouts_by_task.get(task_id) == set(range(samples_per_task))
            for task_id in train_tasks
        )
    )
    group_protocol = all(
        row.get("schema_version") == SCHEMA_VERSION and row.get("record_type") == "group"
        for row in groups
    )

    legal_tasks = True
    target_group_routing = True
    skip_rows_valid = True
    complete_results = []
    complete_groups = []
    skipped_groups = []
    branch_ids: list[str] = []
    candidate_request_ids: list[str] = []

    for row in groups:
        root = root_by_id.get(row.get("root_trajectory_id"))
        task_id = row.get("task_id")
        legal_tasks &= root is not None and task_id in set(train_tasks or [])
        legal_tasks &= root is not None and task_id == root.get("task_id")
        legal_tasks &= row.get("split") == "train"
        selected = task_id in set(target_tasks or [])
        skipped = row.get("skipped") is True
        target_group_routing &= (
            skipped and row.get("skip_reason") == "task_not_selected"
            if not selected
            else row.get("skip_reason") != "task_not_selected"
        )

        if skipped:
            skipped_groups.append(row)
            skip_rows_valid &= (
                isinstance(row.get("skip_reason"), str)
                and bool(row["skip_reason"])
                and row.get("candidates") == []
                and row.get("prefix") is None
                and row.get("effective") is None
                and row.get("cost") is None
            )
            continue

        complete_groups.append(row)
        target_group_routing &= selected
        legal_tasks &= row.get("captured_anchor_turn") == row.get("requested_anchor_turn")
        legal_tasks &= row.get("candidates_per_prefix") == candidates_per_prefix
        legal_tasks &= row.get("continuations_per_candidate") == continuations_per_candidate
        result = _validate_complete_group(row, candidates_per_prefix, continuations_per_candidate)
        complete_results.append(result)
        for candidate in row.get("candidates", []):
            candidate_request_ids.append(candidate.get("candidate_request_id"))
            branch_ids.extend(
                continuation.get("trajectory_id")
                for continuation in candidate.get("continuations", [])
                if isinstance(continuation, dict)
            )

    selected_root_groups = sum(root_task_counts.get(task_id, 0) for task_id in target_tasks or [])
    target_group_count = sum(
        row.get("task_id") in set(target_tasks or []) for row in groups
    )
    identifiers_unique = (
        all(isinstance(value, str) and value for value in branch_ids + candidate_request_ids)
        and len(branch_ids) == len(set(branch_ids))
        and len(candidate_request_ids) == len(set(candidate_request_ids))
        and not (set(branch_ids) & set(root_ids))
        and not (set(candidate_request_ids) & set(root_ids))
    )
    allowed_api_ids = set(root_ids) | set(branch_ids)
    allowed_audit_ids = allowed_api_ids | set(candidate_request_ids)
    logs_have_known_ids = (
        bool(api_rows)
        and all(row.get("trajectory_id") in allowed_api_ids for row in api_rows)
        and bool(audit_rows)
        and all(row.get("trajectory_id") in allowed_audit_ids for row in audit_rows)
    )
    terminal_behavior = summarize_terminal_behavior(audit_rows, api_rows)

    complete_checks = {
        name: all(result[name] for result in complete_results)
        for name in (
            "branch_shape",
            "binary_rewards",
            "candidate_tokens_fixed",
            "derived_values",
            "cost_totals",
        )
    }
    checks = {
        "input_files_valid": not errors,
        "process_success": (run_dir / "exit_code.txt").exists()
        and (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() == "0",
        "metadata_valid": metadata_valid,
        "backbones_complete": roots_complete,
        "branch_group_coverage": group_coverage,
        "branch_schema": group_protocol,
        "target_group_count": metadata_valid and target_group_count == selected_root_groups,
        "target_group_routing": bool(groups) and target_group_routing,
        "skip_rows_valid": skip_rows_valid,
        "task_ids_valid": legal_tasks,
        "identifiers_unique": identifiers_unique,
        "logs_have_known_trajectory_ids": logs_have_known_ids,
        **complete_checks,
        "api_no_terminal_failure": bool(api_rows)
        and all(row.get("success") is True or row.get("retry") is True for row in api_rows),
        "no_calls_after_successful_terminal": terminal_behavior["passed"],
    }
    engineering_accepted = all(checks.values())

    candidate_groups = [
        candidate for row in complete_groups for candidate in row.get("candidates", [])
    ]
    action_variation_rows = [
        summarize_action_variation(row["candidates"]) for row in complete_groups
    ]
    raw_effective_groups = sum(row["raw_effective"] for row in action_variation_rows)
    effective_groups = sum(row["effective"] for row in action_variation_rows)
    diversity = []
    for row in complete_groups:
        tokens = [tuple(candidate["candidate_token_ids"]) for candidate in row["candidates"]]
        diversity.append(len(set(tokens)) / len(tokens))
    repeat_disagreements = 0
    rewards = []
    suffix_policy_tokens = 0
    suffix_total_tokens = 0
    episode_wall_time_s = 0.0
    for candidate in candidate_groups:
        candidate_rewards, continuations = _candidate_values(candidate)
        rewards.extend(candidate_rewards)
        repeat_disagreements += len(set(candidate_rewards)) > 1
        for continuation in continuations:
            if _is_int(continuation.get("suffix_policy_tokens")):
                suffix_policy_tokens += continuation["suffix_policy_tokens"]
            if _is_int(continuation.get("suffix_total_tokens")):
                suffix_total_tokens += continuation["suffix_total_tokens"]
            if _is_number(continuation.get("wall_time_s")):
                episode_wall_time_s += float(continuation["wall_time_s"])
    candidate_policy_tokens = sum(
        candidate["candidate_policy_tokens"]
        for candidate in candidate_groups
        if _is_int(candidate.get("candidate_policy_tokens"))
    )

    branch_id_set = set(branch_ids)
    branch_api = [row for row in api_rows if row.get("trajectory_id") in branch_id_set]
    branch_audit_ids = branch_id_set | set(candidate_request_ids)
    branch_audit = [row for row in audit_rows if row.get("trajectory_id") in branch_audit_ids]
    job_start = read_epoch(run_dir / "job_started.epoch")
    job_finish = read_epoch(run_dir / "job_finished.epoch")
    job_wall_time_s = (
        job_finish - job_start
        if job_start is not None and job_finish is not None and job_finish >= job_start
        else None
    )

    raw_effective_rate = (
        raw_effective_groups / len(complete_groups) if complete_groups else None
    )
    effective_rate = effective_groups / len(complete_groups) if complete_groups else None
    stage = branch_meta.get("stage")
    signal_eligible = stage == "signal" and engineering_accepted and bool(complete_groups)
    provisional_pass = signal_eligible and effective_rate is not None and effective_rate >= 0.20
    signal_screen = {
        "stage": stage,
        "eligible": signal_eligible,
        "status": (
            "provisional_above_threshold"
            if provisional_pass
            else "provisional_below_threshold"
            if signal_eligible
            else "not_assessed_smoke"
            if stage == "smoke"
            else "not_assessed_no_complete_groups"
            if engineering_accepted
            else "not_assessed_engineering_failure"
        ),
        "provisional_pass": bool(provisional_pass),
        "effective_branch_rate": effective_rate,
        "raw_effective_branch_rate": raw_effective_rate,
        "signal_definition": (
            "Pool all suffix outcomes for candidates with identical candidate_token_ids, "
            "then require different pooled means across at least two distinct actions."
        ),
        "descriptive_threshold": 0.20,
        "statistical_significance_tested": False,
        "note": "The 20% threshold is an engineering screen, not evidence of statistical significance.",
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "engineering_accepted": engineering_accepted,
        "engineering_checks": checks,
        "errors": errors,
        "metadata": branch_meta,
        "observed": {
            "backbones": len(roots),
            "groups": len(groups),
            "target_groups": target_group_count,
            "completed_groups": len(complete_groups),
            "skipped_groups": len(skipped_groups),
            "skip_reasons": dict(sorted(Counter(row.get("skip_reason") for row in skipped_groups).items())),
            "candidate_groups": len(candidate_groups),
            "branch_episodes": len(rewards),
            "reward_success_rate": sum(rewards) / len(rewards) if rewards else None,
        },
        "signal_screen": signal_screen,
        "action_variation": {
            "raw_effective_groups": raw_effective_groups,
            "raw_effective_rate": raw_effective_rate,
            "pooled_effective_groups": effective_groups,
            "pooled_effective_rate": effective_rate,
            "groups_with_duplicate_actions": sum(
                row["unique_actions"] < candidates_per_prefix
                for row in action_variation_rows
            ),
            "definition": "Action identity is exact equality of candidate_token_ids.",
        },
        "split_half_stability": summarize_split_half_stability(
            complete_groups,
            continuations_per_candidate if _is_int(continuations_per_candidate) else 0,
        ),
        "candidate_diversity": {
            "mean_unique_fraction": sum(diversity) / len(diversity) if diversity else None,
            "all_candidates_unique_groups": sum(value == 1.0 for value in diversity),
            "all_candidates_unique_rate": (
                sum(value == 1.0 for value in diversity) / len(diversity) if diversity else None
            ),
        },
        "continuation_variation": {
            "candidate_groups_with_reward_disagreement": repeat_disagreements,
            "reward_disagreement_rate": (
                repeat_disagreements / len(candidate_groups) if candidate_groups else None
            ),
        },
        "tokens": {
            "candidate_policy_tokens": candidate_policy_tokens,
            "suffix_policy_tokens": suffix_policy_tokens,
            "total_policy_tokens": candidate_policy_tokens + suffix_policy_tokens,
            "suffix_total_tokens": suffix_total_tokens,
        },
        "api": {
            "all": summarize_api(api_rows),
            "branches": summarize_api(branch_api),
        },
        "tools": {
            "all": summarize_tools(audit_rows),
            "branches": summarize_tools(branch_audit),
        },
        "terminal_behavior": terminal_behavior,
        "wall_time_s": {
            "job": job_wall_time_s,
            "branch_episode_sum": episode_wall_time_s,
        },
    }
