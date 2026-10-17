"""Completeness must follow task IDs and repeats, including zero-reward results."""

import json
import runpy
from pathlib import Path

import pytest

summary_module = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "scripts/eval/summarize_qwen3.py")
)
build_summary = summary_module["build_summary"]


def write_run(tmp_path, scores):
    meta = {
        "protocol": "v2",
        "mode": "eval",
        "eval_split": "train",
        "train_task_ids": [1, 4],
        "test_task_ids": [8],
        "samples_per_task": 4,
        "expected_trajectories": 8,
    }
    (tmp_path / "run.json").write_text(json.dumps(meta))
    rows = [
        {
            "trajectory_id": f"t-{i}",
            "task_id": 1 if i < 4 else 4,
            "split": "train",
            "score": score,
            "schema_verified": True,
            "protocol": "v2",
        }
        for i, score in enumerate(scores)
    ]
    audit = [
        {"event": "generation", "trajectory_id": row["trajectory_id"]} for row in rows
    ]
    for name, values in (
        ("eval_trajectories", rows),
        ("tool_audit", audit),
        ("api", [{"success": True}]),
        ("metrics", [{"step": 0}]),
    ):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    (tmp_path / "exit_code.txt").write_text("0\n")


def test_macros_and_reliability_use_configured_counts(tmp_path):
    write_run(tmp_path, [1, 1, 1, 1, 0, 0, 0, 0])
    result, rows = build_summary(tmp_path)
    assert result["accepted"]
    assert result["splits"]["train"]["success_rate"] == 0.5
    assert result["splits"]["train"]["pass_all_4"] == 0.5
    assert result["splits"]["train"]["pass_at_4"] == 0.5
    assert len(rows) == 2


def test_missing_and_duplicate_samples_rejected(tmp_path):
    write_run(tmp_path, [0] * 7)
    assert not build_summary(tmp_path)[0]["accepted"]
    path = tmp_path / "eval_trajectories.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    assert not build_summary(tmp_path)[0]["accepted"]


def test_holdout_and_nonbinary_scores_rejected(tmp_path):
    write_run(tmp_path, [0] * 8)
    path = tmp_path / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(task_id=8, split="test", score=0.5)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checks = build_summary(tmp_path)[0]["checks"]
    assert not checks["task_split_matches"] and not checks["scores_binary_finite"]


def test_periodic_eval_and_external_checkpoints(tmp_path):
    """Train and validation at the same step must not form one GRPO group."""
    write_run(tmp_path, [0] * 8)
    meta = json.loads((tmp_path / "run.json").read_text())
    checkpoint_root = tmp_path / "archive"
    meta.update(
        mode="train",
        train_batch_size=2,
        total_steps=2,
        start_step=0,
        expected_trajectories=24,
        evaluation_steps=[2],
        eval_samples_per_task=4,
        checkpoint_steps=[2],
        checkpoint_root=str(checkpoint_root),
    )
    (tmp_path / "run.json").write_text(json.dumps(meta))
    rows = []
    for step, validate in [(1, False), (2, False), (2, True)]:
        for task in [1, 4]:
            for sample in range(4):
                rows.append(
                    {
                        "trajectory_id": f"{step}-{validate}-{task}-{sample}",
                        "step": step,
                        "validate": validate,
                        "task_id": task,
                        "split": "train",
                        "score": int(validate),
                        "protocol": "v2",
                        "schema_verified": True,
                    }
                )
    for name, values in [
        ("eval_trajectories", rows),
        (
            "tool_audit",
            [
                {"event": "generation", "trajectory_id": r["trajectory_id"]}
                for r in rows
            ],
        ),
        ("metrics", [{"step": i, "data": {"actor/grad_norm": 0.1}} for i in [1, 2]]),
    ]:
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in values)
        )
    for name in [
        "data.pt",
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "actor/lora_adapter/adapter_config.json",
        "actor/lora_adapter/adapter_model.safetensors",
    ]:
        p = checkpoint_root / "global_step_2" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["splits"]["train"]["success_rate"] == 0
    assert result["evaluations"]["2"]["success_rate"] == 1
    path = tmp_path / "eval_trajectories.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    assert not build_summary(tmp_path)[0]["accepted"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dynamic_sampling_counts_filtered_rollouts_and_actual_final_step(tmp_path):
    write_run(tmp_path, [0] * 8)
    meta = json.loads((tmp_path / "run.json").read_text())
    meta.update(
        mode="train",
        train_batch_size=2,
        samples_per_task=2,
        total_steps=10,
        start_step=0,
        expected_trajectories=12,
        expected_training_trajectories=12,
        evaluation_steps=[],
        evaluation_frequency=10,
        eval_samples_per_task=2,
        checkpoint_steps=[],
        checkpoint_frequency=1,
        checkpoint_root=str(tmp_path / "checkpoints"),
        dynamic_sampling={
            "enabled": True,
            "metric": "outcome_reward",
            "max_num_gen_batches": 2,
            "trajectory_budget": 12,
            "initial_generated_trajectories": 0,
            "initial_optimizer_updates": 0,
            "initial_skipped_updates": 0,
        },
    )
    (tmp_path / "run.json").write_text(json.dumps(meta))

    rows = []
    scores = {
        1: {1: [0, 1], 4: [0, 1]},
        2: {1: [0, 0, 0, 0], 4: [1, 1, 1, 1]},
    }
    for step, task_scores in scores.items():
        for task_id, values in task_scores.items():
            for sample, score in enumerate(values):
                rows.append(
                    {
                        "trajectory_id": f"train-{step}-{task_id}-{sample}",
                        "step": step,
                        "validate": False,
                        "task_id": task_id,
                        "split": "train",
                        "score": score,
                        "protocol": "v2",
                        "schema_verified": True,
                    }
                )
    for task_id in (1, 4):
        for sample in range(2):
            rows.append(
                {
                    "trajectory_id": f"eval-2-{task_id}-{sample}",
                    "step": 2,
                    "validate": True,
                    "task_id": task_id,
                    "split": "train",
                    "score": 1,
                    "protocol": "v2",
                    "schema_verified": True,
                }
            )
    dynamic_metrics = [
        {
            "step": 1,
            "data": {
                "actor/grad_norm": 0.1,
                "dynamic_sampling/generated_trajectories": 4,
                "dynamic_sampling/generated_this_step": 4,
                "dynamic_sampling/num_gen_batches": 1,
                "dynamic_sampling/generated_groups": 2,
                "dynamic_sampling/mixed_groups": 2,
                "dynamic_sampling/retained_groups": 2,
                "dynamic_sampling/dropped_excess_groups": 0,
                "dynamic_sampling/optimizer_updates": 1,
                "dynamic_sampling/skipped_updates": 0,
                "dynamic_sampling/budget_exhausted": 0,
            },
        },
        {
            "step": 2,
            "data": {
                "dynamic_sampling/generated_trajectories": 12,
                "dynamic_sampling/generated_this_step": 8,
                "dynamic_sampling/num_gen_batches": 2,
                "dynamic_sampling/generated_groups": 4,
                "dynamic_sampling/mixed_groups": 0,
                "dynamic_sampling/retained_groups": 0,
                "dynamic_sampling/dropped_excess_groups": 0,
                "dynamic_sampling/optimizer_updates": 1,
                "dynamic_sampling/skipped_updates": 1,
                "dynamic_sampling/budget_exhausted": 1,
            },
        },
    ]
    for name, values in (
        ("eval_trajectories", rows),
        (
            "tool_audit",
            [
                {"event": "generation", "trajectory_id": row["trajectory_id"]}
                for row in rows
            ],
        ),
        ("api", [{"success": True}]),
        ("metrics", dynamic_metrics),
    ):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    for step, state in (
        (1, (4, 1, 0)),
        (2, (12, 1, 1)),
    ):
        checkpoint = tmp_path / f"checkpoints/global_step_{step}"
        for name in (
            "data.pt",
            "actor/model_world_size_1_rank_0.pt",
            "actor/optim_world_size_1_rank_0.pt",
            "actor/extra_state_world_size_1_rank_0.pt",
            "actor/lora_adapter/adapter_config.json",
            "actor/lora_adapter/adapter_model.safetensors",
        ):
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        generated, updates, skips = state
        (checkpoint / "dynamic_sampling.json").write_text(
            json.dumps(
                {
                    "generated_trajectories": generated,
                    "optimizer_updates": updates,
                    "skipped_updates": skips,
                    "global_step": step,
                }
            )
        )

    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["learning"]["groups"] == 6
    assert result["learning"]["mixed_outcome_groups"] == 2
    assert result["learning"]["optimizer_updates"] == 1
    assert result["learning"]["skipped_updates"] == 1
    assert set(result["evaluations"]) == {"2"}

    path = tmp_path / "eval_trajectories.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-5] + rows[-4:]))
    assert not build_summary(tmp_path)[0]["checks"]["dynamic_all_rollouts_recorded"]


def write_joint_run(tmp_path, mode="train"):
    write_run(tmp_path, [0] * 8)
    meta = json.loads((tmp_path / "run.json").read_text())
    meta.update(
        config_name="qwen3_prm_lite_lata",
        reward_mode="prm_lite",
        adv_estimator="grpo_lata",
        lata_alpha=1.05,
        prm_coefficient=0.3,
    )
    rows_path = tmp_path / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    process_scores = [-0.5, -0.25, 0.0, 0.25] * 2
    for row, process_score in zip(rows, process_scores, strict=True):
        row.update(
            process_score=process_score,
            reward_mode="prm_lite",
            training_reward=row["score"]
            if mode == "eval"
            else row["score"] + 0.3 * process_score,
        )
    if mode == "train":
        meta.update(
            mode="train",
            train_batch_size=2,
            total_steps=1,
            start_step=0,
            expected_trajectories=8,
            evaluation_steps=[],
            checkpoint_steps=[],
        )
        for row in rows:
            row.update(step=1, validate=False)
        (tmp_path / "metrics.jsonl").write_text(
            json.dumps({"step": 1, "data": {"actor/grad_norm": 0.1}}) + "\n"
        )
    (tmp_path / "run.json").write_text(json.dumps(meta))
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_joint_shaped_training_reward_keeps_binary_metrics_and_grouping(tmp_path):
    write_joint_run(tmp_path)
    result, rows = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["splits"]["train"]["success_rate"] == 0
    assert {row["c"] for row in rows} == {0}
    assert result["learning"]["groups"] == 2
    assert result["learning"]["mixed_outcome_groups"] == 0
    assert result["learning"]["saturated_outcome_groups_with_process_signal"] == 2
    assert result["learning"]["signal_observed"]


def test_joint_validation_reward_stays_binary(tmp_path):
    write_joint_run(tmp_path, mode="eval")
    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["checks"]["training_reward_formula"]
    path = tmp_path / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["training_reward"] = rows[0]["score"] + 0.3 * rows[0]["process_score"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not build_summary(tmp_path)[0]["checks"]["training_reward_formula"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_score", None),
        ("process_score", 0.6),
        ("training_reward", None),
        ("training_reward", 0.9),
    ],
)
def test_joint_missing_or_invalid_reward_fields_rejected(tmp_path, field, value):
    write_joint_run(tmp_path)
    path = tmp_path / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if value is None:
        rows[0].pop(field)
    else:
        rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not build_summary(tmp_path)[0]["accepted"]


def test_pure_opd_signal_is_independent_of_outcome_groups(tmp_path):
    write_joint_run(tmp_path)
    meta_path = tmp_path / "run.json"
    meta = json.loads(meta_path.read_text())
    meta["distillation"] = {"actor": {"enabled": True, "coef": 1.0, "rl_coef": 0.0}}
    meta_path.write_text(json.dumps(meta))
    data = {
        "actor/grad_norm": 0.1,
        "actor/opd_loss": -0.2,
        "actor/rl_loss": 0.0,
        "actor/opd_coef": 1.0,
        "actor/rl_coef": 0.0,
        "actor/opd_signal_token_fraction": 0.5,
        "distillation/scored_tokens": 20,
        "distillation/scored_trajectories": 8,
        "distillation/policy_version": 0,
    }
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["learning"]["distillation_signal_observed"]
    assert result["learning"]["mixed_outcome_groups"] == 0
    data["actor/opd_signal_token_fraction"] = 0.0
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    result, _ = build_summary(tmp_path)
    assert not result["learning"][
        "signal_observed"
    ]  # PRM variation does not train pure OPD.
    data.pop("distillation/policy_version")
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    assert not build_summary(tmp_path)[0]["accepted"]


def test_self_distillation_requires_consistent_feedback_coverage(tmp_path):
    write_joint_run(tmp_path)
    path = tmp_path / "run.json"
    meta = json.loads(path.read_text())
    meta["distillation"] = {
        "actor": {"enabled": True, "coef": 1.0, "rl_coef": 0.0},
        "teacher": {
            "teacher_mode": "self_feedback",
            "feedback_mode": "F2",
            "feedback_version": "opsd-f2-v1",
        },
    }
    path.write_text(json.dumps(meta))
    data = {
        "actor/grad_norm": 0.1,
        "actor/opd_loss": -0.2,
        "actor/rl_loss": 0.0,
        "actor/opd_coef": 1.0,
        "actor/rl_coef": 0.0,
        "actor/opd_signal_token_fraction": 0.5,
        "distillation/scored_tokens": 20,
        "distillation/scored_trajectories": 8,
        "distillation/policy_version": 0,
        "distillation/feedback_trajectories": 6,
        "distillation/feedback_tokens": 42,
        "distillation/feedback_coverage": 0.75,
    }
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["learning"]["teacher_mode"] == "self_feedback"
    assert result["learning"]["feedback_version"] == "opsd-f2-v1"
    assert result["learning"]["feedback_by_step"][0]["feedback_coverage"] == 0.75
    data["distillation/feedback_coverage"] = 1.0
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    assert not build_summary(tmp_path)[0]["accepted"]
    data.pop("distillation/feedback_tokens")
    path.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    assert not build_summary(tmp_path)[0]["accepted"]


def test_skill_sd_uses_its_own_loss_keys_and_checks_skill_coverage(tmp_path):
    write_joint_run(tmp_path)
    path = tmp_path / "run.json"
    meta = json.loads(path.read_text())
    meta["distillation"] = {
        "actor": {
            "enabled": True,
            "coef": 0.001,
            "rl_coef": 1.0,
            "loss_type": "skill_sd",
        },
        "teacher": {"teacher_mode": "external", "use_skills": True},
    }
    path.write_text(json.dumps(meta))
    data = {
        "actor/grad_norm": 0.1,
        "actor/skill_sd_loss": 0.2,
        "actor/rl_loss": -0.1,
        "actor/skill_sd_coef": 0.001,
        "actor/rl_coef": 1.0,
        "actor/skill_sd_signal_token_fraction": 0.5,
        "distillation/scored_tokens": 20,
        "distillation/scored_trajectories": 8,
        "distillation/policy_version": 0,
        "distillation/skill_trajectories": 4,
        "distillation/skill_tokens": 200,
        "distillation/skill_coverage": 0.5,
    }
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    result, _ = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["learning"]["distillation_loss_type"] == "skill_sd"
    data["distillation/skill_coverage"] = 1.0
    metrics.write_text(json.dumps({"step": 1, "data": data}) + "\n")
    assert not build_summary(tmp_path)[0]["accepted"]


def test_skill_selector_checkpoint_must_match_saved_model_and_bank(tmp_path):
    write_joint_run(tmp_path)
    path = tmp_path / "run.json"
    meta = json.loads(path.read_text())
    meta.update(checkpoint_steps=[1], split_sha256="split-sha")
    meta["distillation"] = {
        "actor": {
            "enabled": True,
            "coef": 0.001,
            "rl_coef": 1.0,
            "loss_type": "skill_sd",
        },
        "teacher": {
            "state_manager": "src.training.distillation.SkillTeacherState",
            "use_skills": True,
            "skill_bank_sha256": "bank-sha",
            "skill_selection": "ucb",
            "ucb_exploration_coef": 2**0.5,
        },
    }
    path.write_text(json.dumps(meta))
    key = "checkpoint_1_skill_selector"
    assert not build_summary(tmp_path)[0]["checks"][key]
    checkpoint = tmp_path / "checkpoints/global_step_1"
    checkpoint.mkdir(parents=True)
    state = {
        "version": "skill-selector-checkpoint-v1",
        "global_step": 1,
        "selector": {
            "last_step": 1,
            "bank_sha256": "bank-sha",
            "split_sha256": "split-sha",
            "mode": "ucb",
            "exploration_coef": 2**0.5,
        },
    }
    file = checkpoint / "skill_selector.json"
    file.write_text(json.dumps(state))
    assert build_summary(tmp_path)[0]["checks"][key]
    state["selector"]["bank_sha256"] = "other-bank"
    file.write_text(json.dumps(state))
    assert not build_summary(tmp_path)[0]["checks"][key]
    state["selector"]["bank_sha256"] = "bank-sha"
    state["global_step"] = 2
    file.write_text(json.dumps(state))
    assert not build_summary(tmp_path)[0]["checks"][key]
