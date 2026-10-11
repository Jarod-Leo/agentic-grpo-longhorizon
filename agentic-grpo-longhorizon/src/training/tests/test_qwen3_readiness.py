"""CPU checks for resumed selector state and its original outcome audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.training.skill_bank import SKILL_VERSION, file_sha256
from src.training.skill_selector import TaskSkillSelector


@pytest.fixture
def readiness():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/train/grpo/check_qwen3_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("qwen3_readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def selector_resume(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"seen_task_ids": [0, 1], "unseen_task_ids": [2]}))
    skills = [
        {
            "task_id": 0,
            "skill_id": skill_id,
            "source_trajectory_ids": [skill_id],
            "success_analysis": "Use current observations.",
            "mistake_analysis": "No mistakes observed.",
            "golden_workflow": "Apply the airline policy before updating the reservation.",
        }
        for skill_id in ["skill-a", "skill-b"]
    ]
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "version": SKILL_VERSION,
                "provenance": {"split_sha256": file_sha256(split)},
                "train_task_ids": [0, 1],
                "test_task_ids": [2],
                "skills": skills,
            }
        )
    )
    bank_sha, split_sha = file_sha256(bank_path), file_sha256(split)
    teacher = {
        "skill_bank_path": str(bank_path),
        "skill_bank_sha256": bank_sha,
        "split_path": str(split),
        "skill_selection": "ucb",
        "ucb_exploration_coef": 2**0.5,
    }
    meta = {
        "distillation": {"teacher": teacher},
        "split_sha256": split_sha,
        "slurm_job_id": "test-job",
    }
    source, restored, run = (
        tmp_path / "global_step_1",
        tmp_path / "global_step_2",
        tmp_path / "resume",
    )
    for path in (source, restored, run):
        path.mkdir()
    selector = TaskSkillSelector({0: skills}, bank_sha, split_sha)
    selector.select([0], 1)
    selector.commit({0: [0] * 8}, 1)
    (source / "skill_selector.json").write_text(
        json.dumps(
            {
                "version": "skill-selector-checkpoint-v1",
                "global_step": 1,
                "selector": selector.state_dict(),
            }
        )
    )
    outcomes = {0: [1] * 4 + [0] * 4, 1: [1] * 8}
    selector.select(outcomes, 2)
    audit = selector.commit(outcomes, 2)
    (restored / "skill_selector.json").write_text(
        json.dumps(
            {
                "version": "skill-selector-checkpoint-v1",
                "global_step": 2,
                "selector": selector.state_dict(),
            }
        )
    )
    (run / "skill_selection.jsonl").write_text(
        "".join(
            json.dumps({**row, "global_step": 2, "slurm_job_id": "test-job"}) + "\n"
            for row in audit
        )
    )
    (run / "eval_trajectories.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "score": score,
                    "validate": False,
                    "step": 2,
                    "split": "train",
                }
            )
            + "\n"
            for task_id, scores in outcomes.items()
            for score in scores
        )
    )
    return source, restored, run, meta


def test_resume_replay_counts_one_group_and_keeps_fallback_uncounted(
    readiness, selector_resume
):
    source, restored, run, meta = selector_resume
    checks = readiness.check_selector_resume(source, restored, run, meta)
    assert checks == {
        "skill_selector_resume_verified": True,
        "skill_selector_reward_groups": 2,
    }
    statistics = json.loads((restored / "skill_selector.json").read_text())["selector"][
        "statistics"
    ]
    assert statistics == {
        "0": {
            "skill-a": {"n": 1, "sum_reward": 0.0},
            "skill-b": {"n": 1, "sum_reward": 0.5},
        }
    }


@pytest.mark.parametrize("tamper", ["state", "audit", "identity"])
def test_resume_replay_rejects_statistics_audit_or_identity_mismatch(
    readiness, selector_resume, tamper
):
    source, restored, run, meta = selector_resume
    if tamper == "audit":
        path = run / "skill_selection.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["reward_mean"] = 1.0
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = (source if tamper == "identity" else restored) / "skill_selector.json"
        state = json.loads(path.read_text())
        if tamper == "identity":
            state["selector"]["split_sha256"] = "wrong"
        else:
            state["selector"]["statistics"]["0"]["skill-b"]["n"] = 8
        path.write_text(json.dumps(state))
    with pytest.raises((AssertionError, ValueError)):
        readiness.check_selector_resume(source, restored, run, meta)
