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
    mode = meta.get("mode")
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
        "expected_unique_trajectories": len(rows) == meta.get("expected_trajectories")
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
    }
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
        steps = set(
            range(meta.get("start_step", 0) + 1, meta.get("total_steps", 0) + 1)
        )
        grouped = defaultdict(list)
        for row in training_rows:
            grouped[row.get("step")].append(row)
        checks["training_steps"] = set(grouped) == steps
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
        actor_metrics = [row for row in metrics if contains_training_metric(row)]
        checks["actor_updates_per_step"] = {
            row.get("step") for row in actor_metrics
        } == steps
        checks["finite_actor_metrics"] = bool(actor_metrics) and all(
            math.isfinite(value)
            for row in actor_metrics
            for value in row.get("data", row).values()
            if isinstance(value, (float, int))
        )
        eval_grouped = defaultdict(list)
        for row in validation_rows:
            eval_grouped[row.get("step")].append(row)
        checks["evaluation_steps"] = set(eval_grouped) == set(
            meta.get("evaluation_steps", [])
        )
        for step, batch in eval_grouped.items():
            counts = Counter(row.get("task_id") for row in batch)
            checks[f"evaluation_{step}_samples"] = set(counts) == set(tasks) and all(
                n == meta.get("eval_samples_per_task", 8) for n in counts.values()
            )
            eval_tasks = per_task_results(batch, task_splits)
            evaluations[str(step)] = aggregate_split(eval_tasks, eval_split)
        for step in meta.get("checkpoint_steps", sorted(steps)):
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
        successes = defaultdict(list)
        for row in training_rows:
            successes[(row.get("step"), row.get("task_id"))].append(row.get("score"))
        mixed = sum(
            any(v == 0 for v in values) and any(v == 1 for v in values)
            for values in successes.values()
        )
        learning = {
            "groups": len(successes),
            "mixed_outcome_groups": mixed,
            "signal_observed": mixed > 0,
        }
        if not mixed:
            learning["note"] = (
                "No mixed-outcome group: zero GRPO advantage is valid; learning signal remains unverified."
            )
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
        "api": summarize_api(api),
        "learning": learning,
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
