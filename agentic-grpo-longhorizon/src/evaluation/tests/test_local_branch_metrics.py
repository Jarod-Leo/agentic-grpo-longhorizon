"""CPU-only tests for experiment-C branch validation and signal metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.local_branch_metrics import (
    build_local_branch_summary,
    summarize_action_variation,
    summarize_split_half_stability,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate(root_id: str, index: int, rewards: list[int]) -> dict:
    token_ids = [100 + index, 200 + index]
    mean_reward = sum(rewards) / len(rewards)
    other_mean = 1.0 - mean_reward
    continuations = [
        {
            "repeat_index": repeat,
            "trajectory_id": f"{root_id}-branch-c{index}-m{repeat}",
            "outcome_reward": reward,
            "termination": "done",
            "assistant_turns": 2,
            "user_turns": 1,
            "tool_calls": 1,
            "suffix_policy_tokens": 5,
            "suffix_total_tokens": 8,
            "wall_time_s": 1.5,
        }
        for repeat, reward in enumerate(rewards)
    ]
    return {
        "candidate_index": index,
        "candidate_request_id": f"{root_id}-branch-c{index}",
        "candidate_token_ids": token_ids,
        "candidate_text": f"candidate {index}",
        "candidate_policy_tokens": len(token_ids),
        "next_state": "interacting",
        "mean_reward": mean_reward,
        "local_advantage": mean_reward - other_mean,
        "continuations": continuations,
    }


def _complete_group(root_id: str, task_id: int, rewards: list[list[int]]) -> dict:
    candidates = [_candidate(root_id, index, values) for index, values in enumerate(rewards)]
    suffix_tokens = sum(
        continuation["suffix_policy_tokens"]
        for candidate in candidates
        for continuation in candidate["continuations"]
    )
    return {
        "schema_version": "local_branch_diagnostic_v1",
        "record_type": "group",
        "root_trajectory_id": root_id,
        "task_id": task_id,
        "split": "train",
        "step": 0,
        "sample_index": 0,
        "rollout_n": 0,
        "requested_anchor_turn": 2,
        "captured_anchor_turn": 2,
        "candidates_per_prefix": 2,
        "continuations_per_candidate": 2,
        "prefix": {
            "assistant_turns": 2,
            "user_turns": 1,
            "response_tokens": 10,
            "policy_tokens": 10,
            "tool_calls": 1,
            "message_count": 5,
        },
        "skipped": False,
        "skip_reason": None,
        "effective": len({sum(values) / len(values) for values in rewards}) > 1,
        "candidates": candidates,
        "cost": {
            "candidate_policy_tokens": sum(
                candidate["candidate_policy_tokens"] for candidate in candidates
            ),
            "suffix_policy_tokens": suffix_tokens,
            "branch_episodes": 4,
            "wall_time_s": 4.0,
        },
    }


def _skip_group(root_id: str, task_id: int, reason: str) -> dict:
    return {
        "schema_version": "local_branch_diagnostic_v1",
        "record_type": "group",
        "root_trajectory_id": root_id,
        "task_id": task_id,
        "split": "train",
        "step": 0,
        "sample_index": 0,
        "rollout_n": 0,
        "requested_anchor_turn": 2,
        "captured_anchor_turn": None,
        "candidates_per_prefix": 2,
        "continuations_per_candidate": 2,
        "skipped": True,
        "skip_reason": reason,
        "prefix": None,
        "effective": None,
        "candidates": [],
        "cost": None,
    }


def _make_run(tmp_path: Path, stage: str = "smoke") -> None:
    run_meta = {
        "mode": "eval",
        "train_task_ids": [0, 1],
        "samples_per_task": 2,
        "expected_trajectories": 4,
    }
    branch_meta = {
        "stage": stage,
        "expected_backbones": 4,
        "task_ids": [0] if stage == "smoke" else [0, 1],
        "candidates": 2,
        "continuations": 2,
        "anchor_turns": [2],
        "adapter_path": "/checkpoint",
        "simulator_model": "mimo-v2.6-pro",
    }
    (tmp_path / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")
    (tmp_path / "branch_run.json").write_text(json.dumps(branch_meta), encoding="utf-8")
    roots = [
        {
            "trajectory_id": f"root-{task}-{sample}",
            "task_id": task,
            "split": "train",
            "score": 0,
            "schema_verified": True,
            "rollout_n": sample,
        }
        for task in (0, 1)
        for sample in (0, 1)
    ]
    groups = []
    for root in roots:
        root_id, task_id = root["trajectory_id"], root["task_id"]
        if task_id not in branch_meta["task_ids"]:
            group = _skip_group(root_id, task_id, "task_not_selected")
        elif root_id.endswith("-0"):
            group = _complete_group(root_id, task_id, [[0, 0], [1, 1]])
        else:
            group = _complete_group(root_id, task_id, [[0, 1], [0, 1]])
        group["rollout_n"] = root["rollout_n"]
        groups.append(group)
    branch_ids = [
        continuation["trajectory_id"]
        for group in groups
        for candidate in group["candidates"]
        for continuation in candidate["continuations"]
    ]
    candidate_ids = [
        candidate["candidate_request_id"]
        for group in groups
        for candidate in group["candidates"]
    ]
    _write_jsonl(tmp_path / "eval_trajectories.jsonl", roots)
    _write_jsonl(tmp_path / "branches/worker-1.jsonl", groups)
    _write_jsonl(
        tmp_path / "api.jsonl",
        [
            {
                "trajectory_id": trajectory_id,
                "success": True,
                "retry": False,
                "attempt": 1,
                "started_at": index,
                "finished_at": index + 0.5,
                "latency_s": 0.5,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
            for index, trajectory_id in enumerate([root["trajectory_id"] for root in roots] + branch_ids)
        ],
    )
    _write_jsonl(
        tmp_path / "tool_audit.jsonl",
        [
            {"event": "generation", "trajectory_id": trajectory_id, "tool_tag_starts": 0, "calls": []}
            for trajectory_id in [root["trajectory_id"] for root in roots] + candidate_ids + branch_ids
        ],
    )
    (tmp_path / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "job_started.epoch").write_text("10\n", encoding="utf-8")
    (tmp_path / "job_finished.epoch").write_text("20\n", encoding="utf-8")


def test_smoke_is_engineering_accepted_but_never_signal_pass(tmp_path):
    _make_run(tmp_path, stage="smoke")
    summary = build_local_branch_summary(tmp_path)
    assert summary["engineering_accepted"], summary["engineering_checks"]
    assert summary["signal_screen"]["status"] == "not_assessed_smoke"
    assert summary["signal_screen"]["provisional_pass"] is False
    assert summary["observed"]["backbones"] == 4
    assert summary["observed"]["target_groups"] == 2
    assert summary["api"]["all"]["requests"] == 12
    assert summary["api"]["branches"]["requests"] == 8
    assert summary["tokens"]["candidate_policy_tokens"] == 8
    assert summary["tokens"]["suffix_policy_tokens"] == 40


def test_signal_screen_is_provisional_and_reports_continuation_noise(tmp_path):
    _make_run(tmp_path, stage="signal")
    summary = build_local_branch_summary(tmp_path)
    assert summary["engineering_accepted"], summary["engineering_checks"]
    assert summary["signal_screen"]["status"] == "provisional_above_threshold"
    assert summary["signal_screen"]["statistical_significance_tested"] is False
    assert summary["signal_screen"]["effective_branch_rate"] == 0.5
    assert summary["signal_screen"]["raw_effective_branch_rate"] == 0.5
    assert summary["continuation_variation"]["reward_disagreement_rate"] == 0.5
    assert summary["candidate_diversity"]["mean_unique_fraction"] == 1.0


def test_identical_actions_pool_suffix_noise_without_forcing_diversity(tmp_path):
    _make_run(tmp_path, stage="signal")
    path = tmp_path / "branches/worker-1.jsonl"
    groups = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for group in groups:
        for candidate in group["candidates"]:
            candidate["candidate_token_ids"] = [7, 8]
    _write_jsonl(path, groups)

    summary = build_local_branch_summary(tmp_path)

    assert summary["engineering_accepted"], summary["engineering_checks"]
    assert summary["signal_screen"]["raw_effective_branch_rate"] == 0.5
    assert summary["signal_screen"]["effective_branch_rate"] == 0.0
    assert summary["signal_screen"]["status"] == "provisional_below_threshold"
    assert summary["action_variation"]["groups_with_duplicate_actions"] == 4


def test_duplicate_candidates_receive_the_same_pooled_action_mean():
    candidates = [
        _candidate("root", index, rewards)
        for index, rewards in enumerate(([0, 0], [0, 1], [0, 0], [0, 0]))
    ]
    for candidate in candidates:
        candidate["candidate_token_ids"] = [7, 8]

    variation = summarize_action_variation(candidates)

    assert variation["raw_candidate_mean_rewards"] == [0.0, 0.5, 0.0, 0.0]
    assert variation["pooled_candidate_mean_rewards"] == [0.125] * 4
    assert variation["raw_effective"] is True
    assert variation["effective"] is False


def test_split_half_stability_reports_same_best_switches_and_ties():
    same = {
        "candidates": [
            _candidate("same", 0, [1, 1, 1, 1]),
            _candidate("same", 1, [0, 0, 0, 0]),
        ]
    }
    switched = {
        "candidates": [
            _candidate("switched", 0, [1, 1, 0, 0]),
            _candidate("switched", 1, [0, 0, 1, 1]),
        ]
    }
    tied = {
        "candidates": [
            _candidate("tied", 0, [1, 1, 1, 1]),
            _candidate("tied", 1, [1, 1, 0, 0]),
            _candidate("tied", 2, [0, 0, 0, 0]),
        ]
    }

    stability = summarize_split_half_stability([same, switched, tied], 4)

    assert stability["available"] is True
    assert stability["multi_action_prefixes"] == 3
    assert stability["informative_in_both_halves"] == 3
    assert stability["unique_best_in_both_halves"] == 2
    assert stability["ties_in_either_half"] == 1
    assert stability["same_unique_best_action"] == 1
    assert stability["same_unique_best_action_rate"] == 0.5


def test_post_terminal_generation_rejects_engineering_gate(tmp_path):
    _make_run(tmp_path, stage="signal")
    trajectory_id = "root-0-0"
    audit_path = tmp_path / "tool_audit.jsonl"
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    audit_rows.extend(
        [
            {
                "event": "execution",
                "trajectory_id": trajectory_id,
                "name": "transfer_to_human_agents",
                "error": False,
                "time": 100.0,
            },
            {"event": "generation", "trajectory_id": trajectory_id, "time": 101.0},
        ]
    )
    _write_jsonl(audit_path, audit_rows)

    summary = build_local_branch_summary(tmp_path)

    assert summary["engineering_accepted"] is False
    assert summary["engineering_checks"]["no_calls_after_successful_terminal"] is False
    assert summary["terminal_behavior"]["post_terminal_generation_events"] == 1


@pytest.mark.parametrize("corruption", ["missing_repeat", "nonbinary", "duplicate_id", "bad_task"])
def test_engineering_corruption_is_rejected(tmp_path, corruption):
    _make_run(tmp_path, stage="signal")
    path = tmp_path / "branches/worker-1.jsonl"
    groups = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if corruption == "missing_repeat":
        groups[0]["candidates"][0]["continuations"].pop()
    elif corruption == "nonbinary":
        groups[0]["candidates"][0]["continuations"][0]["outcome_reward"] = 0.5
    elif corruption == "duplicate_id":
        groups[0]["candidates"][1]["continuations"][0]["trajectory_id"] = (
            groups[0]["candidates"][0]["continuations"][0]["trajectory_id"]
        )
    else:
        groups[0]["task_id"] = 99
    _write_jsonl(path, groups)

    summary = build_local_branch_summary(tmp_path)
    assert summary["engineering_accepted"] is False
    assert summary["signal_screen"]["status"] == "not_assessed_engineering_failure"
