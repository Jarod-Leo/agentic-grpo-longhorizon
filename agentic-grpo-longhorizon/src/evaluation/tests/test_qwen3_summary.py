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


def test_branch_audit_allows_only_validated_auxiliary_ids(tmp_path, monkeypatch):
    write_run(tmp_path, [0] * 8)
    path = tmp_path / "run.json"
    meta = json.loads(path.read_text())
    meta["local_branch"] = {
        "enable": True, "trigger": "critical_prefix", "candidates": 4, "continuations": 8,
    }
    path.write_text(json.dumps(meta))
    monkeypatch.setitem(build_summary.__globals__, "summarize_training_branches", lambda *a, **k: {
        "engineering_accepted": True,
        "allowed_auxiliary_trajectory_ids": ["t-0-branch-c1-m0"],
    })
    audit = tmp_path / "tool_audit.jsonl"
    with audit.open("a") as stream:
        stream.write(json.dumps({"event": "generation", "trajectory_id": "t-0-branch-c1-m0"}) + "\n")
    assert build_summary(tmp_path)[0]["accepted"]
    with audit.open("a") as stream:
        stream.write(json.dumps({"event": "generation", "trajectory_id": "unexplained"}) + "\n")
    assert not build_summary(tmp_path)[0]["checks"]["audit_covers_trajectories"]


def test_teacher_and_user_api_costs_are_separate(tmp_path):
    write_run(tmp_path, [0] * 8)
    api = [{"success": True, "role": role} for role in ("teacher", "teacher", "user")]
    (tmp_path / "api.jsonl").write_text("".join(json.dumps(row) + "\n" for row in api))
    result = build_summary(tmp_path)[0]
    assert set(result["api_by_role"]) == {"teacher", "user"}
    assert result["accepted"]


def write_handled_teacher_truncation(tmp_path):
    write_run(tmp_path, [0] * 8)
    meta_path = tmp_path / "run.json"
    meta = json.loads(meta_path.read_text())
    meta["verified_sft_collection"] = {
        "mode": "teacher",
        "task_ids": [1, 4],
        "attempts_per_task": 4,
        "fixed_budget": True,
    }
    meta_path.write_text(json.dumps(meta))

    rows_path = tmp_path / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0].update(
        collection_failure="teacher_output_truncated",
        termination="teacher_output_truncated",
    )
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    audit_path = tmp_path / "tool_audit.jsonl"
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
    audit = [row for row in audit if row.get("trajectory_id") != "t-0"]
    audit.append(
        {
            "event": "teacher_rejected",
            "trajectory_id": "t-0",
            "reason": "teacher_output_truncated",
        }
    )
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in audit))

    api = [
        {
            "trajectory_id": "t-0",
            "role": "teacher",
            "attempt": 1,
            "max_completion_tokens": 2048,
            "teacher_rejection_scope": "verified_sft_collection",
            "success": False,
            "retry": True,
            "response_received": True,
            "finish_reason": "length",
            "error_type": "MimoResponseError",
            "error_reason": "teacher_output_truncated",
        },
        {
            "trajectory_id": "t-0",
            "role": "teacher",
            "attempt": 2,
            "max_completion_tokens": 4096,
            "teacher_rejection_scope": "verified_sft_collection",
            "success": False,
            "retry": False,
            "response_received": True,
            "finish_reason": "length",
            "error_type": "TeacherOutputTruncatedError",
            "error_reason": "teacher_output_truncated",
        },
    ]
    (tmp_path / "api.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in api)
    )


def test_verified_collection_accepts_audited_teacher_truncation(tmp_path):
    write_handled_teacher_truncation(tmp_path)
    result, task_rows = build_summary(tmp_path)
    assert result["accepted"], result["checks"]
    assert result["observed_trajectories"] == 8
    assert result["api"]["failed_attempts"] == 2
    assert result["api"]["handled_teacher_output_truncations"] == 1
    assert result["api"]["teacher_output_truncation_attempts"] == 2
    assert result["verified_sft_collection"] == {
        "completion_accepted": True,
        "attempts_recorded": 8,
        "environment_result_attempts": 7,
        "collection_failures": {"teacher_output_truncated": 1},
        "handled_teacher_output_truncations": 1,
        "sft_data_acceptance": "reported_by_build_verified_sft",
    }
    assert sum(row["n"] for row in task_rows) == 7


@pytest.mark.parametrize(
    "terminal_update",
    [
        {"role": "user"},
        {"teacher_rejection_scope": None},
        {
            "error_type": "AuthenticationError",
            "error_reason": "AuthenticationError",
            "response_received": False,
            "finish_reason": None,
        },
        {
            "error_type": "InternalServerError",
            "error_reason": "InternalServerError",
            "response_received": False,
            "finish_reason": None,
        },
        {"error_type": "RuntimeError", "error_reason": "RuntimeError"},
    ],
)
def test_verified_collection_does_not_exempt_other_terminal_api_failures(
    tmp_path, terminal_update
):
    write_handled_teacher_truncation(tmp_path)
    path = tmp_path / "api.jsonl"
    api = [json.loads(line) for line in path.read_text().splitlines()]
    api[-1].update(terminal_update)
    path.write_text("".join(json.dumps(row) + "\n" for row in api))
    result = build_summary(tmp_path)[0]
    assert not result["checks"]["api_no_terminal_failure"]
    assert not result["accepted"]


def test_verified_collection_requires_matching_teacher_rejection_audit(tmp_path):
    write_handled_teacher_truncation(tmp_path)
    path = tmp_path / "tool_audit.jsonl"
    audit = [json.loads(line) for line in path.read_text().splitlines()]
    audit[-1]["trajectory_id"] = "other-trajectory"
    path.write_text("".join(json.dumps(row) + "\n" for row in audit))
    result = build_summary(tmp_path)[0]
    assert not result["checks"]["api_no_terminal_failure"]


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

    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta["expected_trajectories"] = None
    meta["expected_training_trajectories"] = None
    meta["dynamic_sampling"].update(
        trajectory_budget=0,
        target_optimizer_updates=2,
        stop_condition="target_optimizer_updates",
    )
    (tmp_path / "run.json").write_text(json.dumps(meta))
    target_result, _ = build_summary(tmp_path)
    assert not target_result["checks"]["dynamic_target_optimizer_updates"]


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
