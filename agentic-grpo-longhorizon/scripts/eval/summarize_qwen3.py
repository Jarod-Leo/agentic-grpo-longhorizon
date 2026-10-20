"""Validate shared Qwen3 training/evaluation artifacts without fixed task counts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.run_metrics import (
    aggregate_split,
    contains_training_metric,
    is_binary_score,
    per_task_results,
    read_epoch,
    read_json,
    read_jsonl,
    summarize_api,
    summarize_tools,
)
from src.evaluation.terminal_audit import summarize_terminal_behavior
from src.evaluation.local_branching import (
    read_training_branch_records,
    summarize_training_branches,
)


def finite_number(value) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def build_summary(run: Path) -> tuple[dict, list[dict]]:
    errors = []
    meta, err = read_json(run / "run.json", {})
    errors.extend(err)
    inputs = {}
    for name in ("eval_trajectories", "tool_audit", "api", "metrics"):
        inputs[name], err = read_jsonl(run / f"{name}.jsonl")
        errors.extend(err)
    rows, audit, api, metrics = (inputs[name] for name in inputs)
    ids = [row.get("trajectory_id") for row in rows]
    tools = summarize_tools(audit)
    terminal_behavior = summarize_terminal_behavior(audit, api)
    mode = meta.get("mode")
    dynamic = meta.get("dynamic_sampling", {})
    dynamic_enabled = mode == "train" and dynamic.get("enabled") is True
    eval_split = meta.get("eval_split", "train")
    tasks = (
        meta.get(f"{eval_split}_task_ids", [])
        if mode == "eval"
        else meta.get("train_task_ids", [])
    )
    task_splits = dict.fromkeys(tasks, eval_split if mode == "eval" else "train")
    training_rows = (
        [row for row in rows if not row.get("validate", False)]
        if mode == "train"
        else rows
    )
    validation_rows = (
        [row for row in rows if row.get("validate", False)] if mode == "train" else []
    )
    task_rows = per_task_results(training_rows, task_splits)
    exit_path = run / "exit_code.txt"
    checks = {
        "process_success": exit_path.exists() and exit_path.read_text().strip() == "0",
        "expected_unique_trajectories": (
            dynamic_enabled or len(rows) == meta.get("expected_trajectories")
        )
        and len(set(ids)) == len(ids)
        and None not in ids,
        "task_split_matches": bool(rows)
        and all(
            row.get("split") == task_splits.get(row.get("task_id")) for row in rows
        ),
        "scores_binary_finite": bool(rows)
        and all(is_binary_score(row.get("score")) for row in rows),
        "input_protocol_verified": bool(rows)
        and all(
            row.get("protocol") == meta.get("protocol")
            and row.get("schema_verified") is True
            for row in rows
        ),
        "thinking_disabled": tools["nonempty_thinking_events"] == 0,
        "audit_covers_trajectories": {
            row.get("trajectory_id")
            for row in audit
            if row.get("event") == "generation"
        }
        == set(ids),
        "api_no_terminal_failure": bool(api)
        and all(row.get("success") or row.get("retry") for row in api),
        "no_calls_after_terminal": terminal_behavior["passed"],
    }
    local_branch = {}
    local_config = meta.get("local_branch", {})
    if local_config.get("enable") and local_config.get("trigger") == "critical_prefix":
        records = read_training_branch_records(run)
        local_branch = summarize_training_branches(
            records,
            rows,
            candidates=int(local_config["candidates"]),
            continuations=int(local_config["continuations"]),
        )
        checks["local_branch_records_valid"] = local_branch["engineering_accepted"]
        generation_ids = {
            row.get("trajectory_id") for row in audit if row.get("event") == "generation"
        }
        allowed_ids = set(local_branch["allowed_auxiliary_trajectory_ids"])
        checks["audit_covers_trajectories"] = (
            set(ids) <= generation_ids <= set(ids) | allowed_ids
        )
    if "reward_mode" in meta:
        reward_mode = meta.get("reward_mode")
        coefficient = meta.get("prm_coefficient")
        adv_estimator = meta.get("adv_estimator")
        lata_alpha = meta.get("lata_alpha")

        def reward_formula_matches(row: dict) -> bool:
            outcome = row.get("score")
            process_score = row.get("process_score")
            training_reward = row.get("training_reward")
            if not all(
                finite_number(value)
                for value in (outcome, process_score, training_reward)
            ):
                return False
            expected = float(outcome)
            is_training = mode == "train" and not row.get("validate", False)
            if is_training and reward_mode == "prm_lite":
                expected += float(coefficient) * float(process_score)
            return math.isclose(
                float(training_reward), expected, rel_tol=1e-6, abs_tol=1e-6
            )

        expected_coefficient = 0.3 if reward_mode == "prm_lite" else 0.0
        checks.update(
            {
                "reward_configuration": reward_mode in {"binary", "prm_lite"}
                and finite_number(coefficient)
                and math.isclose(float(coefficient), expected_coefficient)
                and isinstance(adv_estimator, str)
                and bool(adv_estimator)
                and (adv_estimator != "grpo_lata" or finite_number(lata_alpha)),
                "reward_mode_matches": bool(rows)
                and all(row.get("reward_mode") == reward_mode for row in rows),
                "process_scores_finite_bounded": bool(rows)
                and all(
                    finite_number(row.get("process_score"))
                    and -0.5 <= row["process_score"] <= 0.5
                    for row in rows
                ),
                "training_rewards_finite": bool(rows)
                and all(finite_number(row.get("training_reward")) for row in rows),
                "training_reward_formula": bool(rows)
                and all(reward_formula_matches(row) for row in rows),
            }
        )
    learning = {}
    evaluations = {}
    if mode == "eval":
        counts = Counter(row.get("task_id") for row in rows)
        checks.update(
            {
                "samples_per_task": bool(tasks)
                and set(counts) == set(tasks)
                and all(counts[t] == meta.get("samples_per_task") for t in tasks),
                "validation_only": bool(metrics)
                and all(
                    row.get("step") == 0 and not contains_training_metric(row)
                    for row in metrics
                ),
                "no_checkpoints_written": not list((run / "checkpoints").rglob("*.pt")),
            }
        )
    elif mode == "train":
        dynamic_metrics = [
            row
            for row in metrics
            if "dynamic_sampling/generated_trajectories" in row.get("data", row)
        ]
        final_step = (
            max((row.get("step", 0) for row in dynamic_metrics), default=0)
            if dynamic_enabled
            else meta.get("total_steps", 0)
        )
        steps = set(range(meta.get("start_step", 0) + 1, final_step + 1))
        grouped = defaultdict(list)
        for row in training_rows:
            grouped[row.get("step")].append(row)
        checks["training_steps"] = set(grouped) == steps
        actor_metrics = [row for row in metrics if contains_training_metric(row)]
        if dynamic_enabled:
            metric_by_step = defaultdict(list)
            for row in dynamic_metrics:
                metric_by_step[row.get("step")].append(row)
            ordered_dynamic = [
                metric_by_step[step][0]
                for step in sorted(steps)
                if metric_by_step[step]
            ]
            initial_generated = int(dynamic.get("initial_generated_trajectories", 0))
            initial_updates = int(dynamic.get("initial_optimizer_updates", 0))
            initial_skips = int(dynamic.get("initial_skipped_updates", 0))
            budget = int(dynamic.get("trajectory_budget", 0))
            target_optimizer_updates = int(
                dynamic.get("target_optimizer_updates", 0)
            )
            samples = int(meta.get("samples_per_task", 0))
            batch_trajectories = int(meta.get("train_batch_size", 0)) * samples

            previous_generated = initial_generated
            previous_updates = initial_updates
            previous_skips = initial_skips
            counters_valid = True
            groups_valid = True
            update_steps = set()
            for row in ordered_dynamic:
                data = row.get("data", row)
                generated = data.get("dynamic_sampling/generated_trajectories")
                generated_this_step = data.get("dynamic_sampling/generated_this_step")
                num_batches = data.get("dynamic_sampling/num_gen_batches")
                generated_groups = data.get("dynamic_sampling/generated_groups")
                mixed_groups = data.get("dynamic_sampling/mixed_groups")
                retained_groups = data.get("dynamic_sampling/retained_groups")
                dropped_groups = data.get("dynamic_sampling/dropped_excess_groups")
                optimizer_updates = data.get("dynamic_sampling/optimizer_updates")
                skipped_updates = data.get("dynamic_sampling/skipped_updates")
                values = (
                    generated,
                    generated_this_step,
                    num_batches,
                    generated_groups,
                    mixed_groups,
                    retained_groups,
                    dropped_groups,
                    optimizer_updates,
                    skipped_updates,
                )
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in values
                ):
                    counters_valid = False
                    continue
                counters_valid &= (
                    generated == previous_generated + generated_this_step
                    and optimizer_updates == previous_updates + int(retained_groups > 0)
                    and skipped_updates == previous_skips + int(retained_groups == 0)
                )
                groups_valid &= (
                    generated_this_step == num_batches * batch_trajectories
                    and generated_this_step == generated_groups * samples
                    and 0 <= dropped_groups <= mixed_groups
                    and retained_groups == mixed_groups - dropped_groups
                    and 0 <= retained_groups <= meta.get("train_batch_size", 0)
                )
                if retained_groups > 0:
                    update_steps.add(row.get("step"))
                previous_generated = generated
                previous_updates = optimizer_updates
                previous_skips = skipped_updates

            actor_steps = {row.get("step") for row in actor_metrics}
            final_data = (
                ordered_dynamic[-1].get("data", ordered_dynamic[-1])
                if ordered_dynamic
                else {}
            )
            expected_training_trajectories = meta.get(
                "expected_training_trajectories"
            )
            checks.update(
                {
                    "dynamic_metrics_per_attempt": bool(steps)
                    and set(metric_by_step) == steps
                    and all(len(metric_by_step[step]) == 1 for step in steps),
                    "dynamic_counters_consistent": counters_valid,
                    "dynamic_groups_consistent": groups_valid,
                    "dynamic_all_rollouts_recorded": sum(
                        row.get("data", row).get("dynamic_sampling/generated_this_step", -1)
                        for row in ordered_dynamic
                    )
                    == len(training_rows)
                    and (
                        expected_training_trajectories is None
                        or len(training_rows) == expected_training_trajectories
                    ),
                    "groups_per_step": all(
                        len(grouped[step])
                        == metric_by_step[step][0]
                        .get("data", metric_by_step[step][0])
                        .get("dynamic_sampling/generated_this_step")
                        and all(
                            count % samples == 0
                            for count in Counter(
                                row.get("task_id") for row in grouped[step]
                            ).values()
                        )
                        for step in steps
                    ),
                    "actor_updates_per_step": actor_steps == update_steps,
                }
            )
            if budget > 0:
                checks["dynamic_budget_exhausted"] = (
                    bool(ordered_dynamic)
                    and previous_generated == budget
                    and final_data.get("dynamic_sampling/budget_exhausted") == 1
                    and all(
                        row.get("data", row).get(
                            "dynamic_sampling/budget_exhausted"
                        )
                        == 0
                        for row in ordered_dynamic[:-1]
                    )
                )
            if target_optimizer_updates > 0:
                checks["dynamic_target_optimizer_updates"] = (
                    bool(ordered_dynamic)
                    and previous_updates == target_optimizer_updates
                    and final_data.get("dynamic_sampling/target_optimizer_updates")
                    == target_optimizer_updates
                    and final_data.get("dynamic_sampling/target_reached") == 1
                    and final_data.get("dynamic_sampling/budget_exhausted") == 0
                    and all(
                        row.get("data", row).get(
                            "dynamic_sampling/target_optimizer_updates"
                        )
                        == target_optimizer_updates
                        and row.get("data", row).get(
                            "dynamic_sampling/target_reached"
                        )
                        == 0
                        and row.get("data", row).get(
                            "dynamic_sampling/optimizer_updates", 0
                        )
                        < target_optimizer_updates
                        for row in ordered_dynamic[:-1]
                    )
                )
        else:
            checks["groups_per_step"] = all(
                len(batch) == meta["train_batch_size"] * meta["samples_per_task"]
                and len(Counter(row["task_id"] for row in batch))
                == meta["train_batch_size"]
                and all(
                    n == meta["samples_per_task"]
                    for n in Counter(row["task_id"] for row in batch).values()
                )
                for batch in grouped.values()
            )
            checks["actor_updates_per_step"] = {
                row.get("step") for row in actor_metrics
            } == steps
        checks["finite_actor_metrics"] = (
            bool(actor_metrics) or (dynamic_enabled and not update_steps)
        ) and all(
            math.isfinite(value)
            for row in actor_metrics
            for value in row.get("data", row).values()
            if isinstance(value, (float, int))
        )
        eval_grouped = defaultdict(list)
        for row in validation_rows:
            eval_grouped[row.get("step")].append(row)
        if dynamic_enabled:
            frequency = int(meta.get("evaluation_frequency", -1))
            evaluation_steps = {
                step for step in steps if frequency > 0 and step % frequency == 0
            }
            if final_step:
                evaluation_steps.add(final_step)
        else:
            evaluation_steps = set(meta.get("evaluation_steps", []))
        checks["evaluation_steps"] = set(eval_grouped) == evaluation_steps
        for step, batch in eval_grouped.items():
            counts = Counter(row.get("task_id") for row in batch)
            checks[f"evaluation_{step}_samples"] = set(counts) == set(tasks) and all(
                n == meta.get("eval_samples_per_task", 8) for n in counts.values()
            )
            eval_tasks = per_task_results(batch, task_splits)
            evaluations[str(step)] = aggregate_split(eval_tasks, eval_split)
        if dynamic_enabled:
            frequency = int(meta.get("checkpoint_frequency", -1))
            checkpoint_steps = {
                step for step in steps if frequency > 0 and step % frequency == 0
            }
            if final_step:
                checkpoint_steps.add(final_step)
        else:
            checkpoint_steps = set(meta.get("checkpoint_steps", sorted(steps)))
        for step in sorted(checkpoint_steps):
            checkpoint = (
                Path(meta.get("checkpoint_root", str(run / "checkpoints")))
                / f"global_step_{step}"
            )
            checks[f"checkpoint_{step}_complete"] = all(
                (checkpoint / name).is_file()
                for name in (
                    "data.pt",
                    "actor/model_world_size_1_rank_0.pt",
                    "actor/optim_world_size_1_rank_0.pt",
                    "actor/extra_state_world_size_1_rank_0.pt",
                    "actor/lora_adapter/adapter_config.json",
                    "actor/lora_adapter/adapter_model.safetensors",
                )
            )
            if dynamic_enabled:
                state, err = read_json(checkpoint / "dynamic_sampling.json", {})
                errors.extend(err)
                step_metric = metric_by_step.get(step, [{}])[0]
                step_data = step_metric.get("data", step_metric)
                checks[f"checkpoint_{step}_dynamic_sampling"] = (
                    state.get("global_step") == step
                    and state.get("generated_trajectories")
                    == step_data.get("dynamic_sampling/generated_trajectories")
                    and state.get("optimizer_updates")
                    == step_data.get("dynamic_sampling/optimizer_updates")
                    and state.get("skipped_updates")
                    == step_data.get("dynamic_sampling/skipped_updates")
                )
            distillation = meta.get("distillation", {})
            actor = distillation.get("actor", {})
            teacher = distillation.get("teacher") or {}
            if (
                actor.get("enabled")
                and actor.get("coef", 0) > 0
                and teacher.get("use_skills")
                and teacher.get("state_manager")
            ):
                state, err = read_json(checkpoint / "skill_selector.json", {})
                errors.extend(err)
                selector = state.get("selector", {})
                checks[f"checkpoint_{step}_skill_selector"] = (
                    state.get("version") == "skill-selector-checkpoint-v1"
                    and state.get("global_step") == step
                    and selector.get("last_step") == step
                    and selector.get("bank_sha256") == teacher.get("skill_bank_sha256")
                    and selector.get("split_sha256") == meta.get("split_sha256")
                    and selector.get("mode") == teacher.get("skill_selection")
                    and selector.get("exploration_coef")
                    == teacher.get("ucb_exploration_coef")
                )
        reward_groups = defaultdict(list)
        for row in training_rows:
            reward_groups[(row.get("step"), row.get("task_id"))].append(row)
        mixed = sum(
            any(row.get("score") == 0 for row in group)
            and any(row.get("score") == 1 for row in group)
            for group in reward_groups.values()
        )

        def group_has_training_signal(group: list[dict]) -> bool:
            values = [
                row.get("training_reward")
                if "reward_mode" in meta
                else row.get("score")
                for row in group
            ]
            return (
                all(finite_number(value) for value in values)
                and max(values) - min(values) > 1e-12
            )

        signal_groups = sum(
            group_has_training_signal(group) for group in reward_groups.values()
        )
        saturated_with_process_signal = sum(
            not (
                any(row.get("score") == 0 for row in group)
                and any(row.get("score") == 1 for row in group)
            )
            and group_has_training_signal(group)
            for group in reward_groups.values()
        )
        if dynamic_enabled:
            dynamic_values = [row.get("data", row) for row in ordered_dynamic]
            generated_groups = sum(
                row["dynamic_sampling/generated_groups"] for row in dynamic_values
            )
            mixed = sum(row["dynamic_sampling/mixed_groups"] for row in dynamic_values)
            retained = sum(
                row["dynamic_sampling/retained_groups"] for row in dynamic_values
            )
            dropped = sum(
                row["dynamic_sampling/dropped_excess_groups"] for row in dynamic_values
            )
            learning = {
                "groups": generated_groups,
                "mixed_outcome_groups": mixed,
                "retained_mixed_groups": retained,
                "dropped_excess_mixed_groups": dropped,
                "saturated_outcome_groups": generated_groups - mixed,
                "optimizer_updates": previous_updates - initial_updates,
                "skipped_updates": previous_skips - initial_skips,
                "generated_trajectories": previous_generated - initial_generated,
                "signal_observed": retained > 0,
            }
        else:
            learning = {
                "groups": len(reward_groups),
                "mixed_outcome_groups": mixed,
                "training_reward_signal_groups": signal_groups,
                "saturated_outcome_groups": len(reward_groups) - mixed,
                "saturated_outcome_groups_with_process_signal": saturated_with_process_signal,
                "signal_observed": signal_groups > 0,
            }
        distill = meta.get("distillation", {}).get("actor", {})
        if distill.get("enabled", False) and distill.get("coef", 0) > 0:
            objective = "skill_sd" if distill.get("loss_type") == "skill_sd" else "opd"
            learning["distillation_loss_type"] = distill.get(
                "loss_type", "sampled_token_pg"
            )
            required = {
                f"actor/{objective}_loss",
                "actor/rl_loss",
                f"actor/{objective}_signal_token_fraction",
                f"actor/{objective}_coef",
                "actor/rl_coef",
                "distillation/scored_tokens",
                "distillation/scored_trajectories",
                "distillation/policy_version",
            }
            checks["distillation_metrics_complete"] = bool(actor_metrics) and all(
                required.issubset(row.get("data", {}))
                and all(finite_number(row["data"][key]) for key in required)
                and row["data"]["distillation/scored_tokens"] > 0
                and row["data"]["distillation/scored_trajectories"]
                == meta["train_batch_size"] * meta["samples_per_task"]
                and row["data"]["distillation/policy_version"] == row["step"] - 1
                and 0 <= row["data"][f"actor/{objective}_signal_token_fraction"] <= 1
                and math.isclose(
                    row["data"][f"actor/{objective}_coef"], distill["coef"]
                )
                and math.isclose(row["data"]["actor/rl_coef"], distill["rl_coef"])
                and (distill["rl_coef"] != 0 or row["data"]["actor/rl_loss"] == 0)
                for row in actor_metrics
            )
            teacher_signal = any(
                finite_number(
                    row.get("data", {}).get(f"actor/{objective}_signal_token_fraction")
                )
                and row["data"][f"actor/{objective}_signal_token_fraction"] > 0
                for row in actor_metrics
            )
            teacher = meta.get("distillation", {}).get("teacher") or {}
            learning["teacher_mode"] = teacher.get("teacher_mode", "external")
            if objective == "skill_sd":
                skill_fields = {
                    "distillation/skill_trajectories",
                    "distillation/skill_tokens",
                    "distillation/skill_coverage",
                }
                checks["skill_metrics_complete"] = bool(actor_metrics) and all(
                    skill_fields.issubset(row.get("data", {}))
                    and all(finite_number(row["data"][key]) for key in skill_fields)
                    and 0
                    <= row["data"]["distillation/skill_trajectories"]
                    <= row["data"].get("distillation/scored_trajectories", 0)
                    and row["data"]["distillation/skill_tokens"] >= 0
                    and math.isclose(
                        row["data"]["distillation/skill_coverage"],
                        row["data"]["distillation/skill_trajectories"]
                        / max(
                            row["data"].get("distillation/scored_trajectories", 0), 1
                        ),
                    )
                    for row in actor_metrics
                )
            if teacher.get("teacher_mode") == "self_feedback":
                feedback_fields = {
                    "distillation/feedback_trajectories",
                    "distillation/feedback_tokens",
                    "distillation/feedback_coverage",
                }
                checks["self_feedback_metrics_complete"] = bool(actor_metrics) and all(
                    feedback_fields.issubset(row.get("data", {}))
                    and all(finite_number(row["data"][key]) for key in feedback_fields)
                    and 0
                    <= row["data"]["distillation/feedback_trajectories"]
                    <= row["data"].get("distillation/scored_trajectories", 0)
                    and 0 <= row["data"]["distillation/feedback_tokens"]
                    and math.isclose(
                        row["data"]["distillation/feedback_coverage"],
                        row["data"]["distillation/feedback_trajectories"]
                        / max(
                            row["data"].get("distillation/scored_trajectories", 0), 1
                        ),
                    )
                    for row in actor_metrics
                )
                learning["feedback_version"] = teacher.get("feedback_version")
                learning["feedback_mode"] = teacher.get("feedback_mode")
                learning["feedback_by_step"] = [
                    {
                        "step": row["step"],
                        **{
                            key.split("/")[1]: row.get("data", {}).get(key)
                            for key in sorted(feedback_fields)
                        },
                    }
                    for row in actor_metrics
                ]
            learning["reward_signal_observed"] = signal_groups > 0
            learning["distillation_signal_observed"] = teacher_signal
            learning["signal_observed"] = teacher_signal or (
                distill["rl_coef"] > 0 and signal_groups > 0
            )
        if not learning["signal_observed"]:
            learning["note"] = (
                "No signal observed in enabled training objectives; zero GRPO advantage is valid when group rewards are constant."
            )
        if local_config.get("enable") and local_config.get("emit_training_fields"):
            required = {
                "actor/local_branch_loss",
                "actor/local_branch_coef",
                "actor/local_branch_actions",
                "actor/local_branch_tokens",
                "actor/local_branch_advantage_mean",
            }
            checks["local_branch_actor_metrics"] = bool(actor_metrics) and all(
                required <= row.get("data", {}).keys()
                and all(finite_number(row["data"][key]) for key in required)
                for row in actor_metrics
            )
            learning["reward_signal_observed"] = learning["signal_observed"]
            learning["local_signal_observed"] = local_branch.get("nonzero_signal_groups", 0) > 0
            learning["signal_observed"] |= learning["local_signal_observed"]
            if learning["signal_observed"]:
                learning.pop("note", None)
    else:
        checks["known_mode"] = False
    log = (
        (run / "run.log").read_text(errors="replace")
        if (run / "run.log").exists()
        else ""
    )
    init = re.findall(r"LORA_INITIALIZATION (\{[^\n]*\})", log)
    initializations = [json.loads(text) for text in init]
    start, finish = (
        read_epoch(run / "job_started.epoch"),
        read_epoch(run / "job_finished.epoch"),
    )
    summary = {
        "accepted": not errors and all(checks.values()),
        "metadata": meta,
        "checks": checks,
        "errors": errors,
        "observed_trajectories": len(rows),
        "splits": {
            name: aggregate_split(task_rows, name)
            for name in sorted(set(task_splits.values()))
        },
        "tools": tools,
        "terminal_behavior": terminal_behavior,
        "api": summarize_api(api),
        "api_by_role": {
            role: summarize_api([row for row in api if row.get("role", "user") == role])
            for role in sorted({row.get("role", "user") for row in api})
        },
        "learning": learning,
        "local_branch": local_branch,
        "evaluations": evaluations,
        "lora_initialization": initializations,
        "termination_reasons": dict(Counter(row.get("termination") for row in rows)),
        "gpu_hours": (finish - start) / 3600
        if start is not None and finish is not None
        else None,
    }
    return summary, task_rows


def write_outputs(run: Path, summary: dict, task_rows: list[dict]) -> None:
    (run / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (run / "task_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_rows)
    )

    def pct(value):
        return "—" if value is None else f"{value:.2%}"

    lines = [
        "# Qwen3 实验结果",
        "",
        f"验收通过：{summary['accepted']}；协议：{summary['metadata'].get('protocol')}。",
        "",
        "| 划分 | 任务数 | 轨迹数 | Pass@1（宏平均成功率） | Pass^4（四次全成功） | Pass@4（至少一次成功） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["splits"].items():
        lines.append(
            f"| {name} | {row['tasks']} | {row['n']} | {pct(row['success_rate'])} | {pct(row['pass_all_4'])} | {pct(row['pass_at_4'])} |"
        )
    for step, row in summary.get("evaluations", {}).items():
        lines.append(
            f"| train 独立评测 step {step} | {row['tasks']} | {row['n']} | {pct(row['success_rate'])} | {pct(row['pass_all_4'])} | {pct(row['pass_at_4'])} |"
        )
    lines += [
        "",
        "成功率指标仅使用二元 outcome；shaped training_reward 仅用于训练与学习信号诊断。",
        "训练轨迹的指标仅用于诊断；模型效果以固定 checkpoint 的独立评测为准。未完成运行不构成可比较结果。",
        "",
        f"失败检查：{[key for key, ok in summary['checks'].items() if not ok]}。",
        f"读取错误：{summary['errors']}。",
        "",
    ]
    if summary["learning"]:
        lines.append(f"学习信号：{summary['learning']}。\n")
    (run / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary, tasks = build_summary(args.run)
    write_outputs(args.run, summary, tasks)
    print(
        f"RUN_SUMMARY accepted={summary['accepted']} path={args.run / 'summary.json'}"
    )
    raise SystemExit(0 if summary["accepted"] else 1)


if __name__ == "__main__":
    main()
