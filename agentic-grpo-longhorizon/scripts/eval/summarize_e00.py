"""Validate and summarize an E00 Qwen3-8B baseline evaluation run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_SAMPLES_PER_TASK = 8
REGRESSION_TASK_IDS = (11, 18, 28, 40)


from src.evaluation.run_metrics import (  # noqa: E402
    read_json,
    read_jsonl,
    mean,
    percentile,
    as_timestamp,
    read_epoch,
    is_binary_score,
    per_task_results,
    aggregate_split,
    summarize_tools,
    summarize_api,
    contains_training_metric,
    checkpoint_files,
)


def build_summary(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    split, new_errors = read_json(run / "split.json", {})
    errors.extend(new_errors)
    raw_trajectories, new_errors = read_jsonl(run / "eval_trajectories.jsonl")
    errors.extend(new_errors)
    audit_rows, new_errors = read_jsonl(run / "tool_audit.jsonl")
    errors.extend(new_errors)
    api_rows, new_errors = read_jsonl(run / "api.jsonl")
    errors.extend(new_errors)
    metrics_rows, new_errors = read_jsonl(run / "metrics.jsonl")
    errors.extend(new_errors)

    train_ids = split.get("train_task_ids", [])
    dev_ids = split.get("dev_task_ids", [])
    test_ids = split.get("test_task_ids", [])
    task_splits = {task_id: "train" for task_id in train_ids} | {
        task_id: "dev" for task_id in dev_ids
    }

    seen_trajectory_ids: set[Any] = set()
    duplicate_ids: list[Any] = []
    unique_trajectories = []
    for row in raw_trajectories:
        trajectory_id = row.get("trajectory_id")
        if trajectory_id in seen_trajectory_ids:
            duplicate_ids.append(trajectory_id)
            continue
        seen_trajectory_ids.add(trajectory_id)
        unique_trajectories.append(row)

    task_rows = per_task_results(unique_trajectories, task_splits)
    expected_tasks = set(train_ids) | set(dev_ids)
    observed_counts = Counter(
        row.get("task_id")
        for row in unique_trajectories
        if row.get("task_id") in expected_tasks
        and row.get("split") == task_splits.get(row.get("task_id"))
    )

    exit_code = None
    exit_path = run / "exit_code.txt"
    if exit_path.exists():
        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except ValueError:
            errors.append("invalid:exit_code.txt")
    else:
        errors.append("missing:exit_code.txt")

    checkpoints = checkpoint_files(run)
    tools = summarize_tools(audit_rows)
    starts_and_finishes = [
        (as_timestamp(row.get("started_at")), as_timestamp(row.get("finished_at")))
        for row in unique_trajectories
    ]
    latencies = [
        finished - started
        for started, finished in starts_and_finishes
        if started is not None and finished is not None
    ]
    assistant_tokens = [
        float(row["assistant_tokens"])
        for row in unique_trajectories
        if isinstance(row.get("assistant_tokens"), (int, float))
        and not isinstance(row["assistant_tokens"], bool)
    ]

    checks = {
        "split_30_10_10": len(train_ids) == 30
        and len(dev_ids) == 10
        and len(test_ids) == 10,
        "process_success": exit_code == 0,
        "trajectories_320_unique": len(raw_trajectories) == 320
        and len(unique_trajectories) == 320
        and not duplicate_ids,
        "eight_per_train_dev_task": set(observed_counts) == expected_tasks
        and all(
            observed_counts[task_id] == EXPECTED_SAMPLES_PER_TASK
            for task_id in expected_tasks
        ),
        "no_test_trajectories": all(
            row.get("task_id") not in set(test_ids) and row.get("split") != "test"
            for row in raw_trajectories
        ),
        "trajectory_split_matches": all(
            row.get("task_id") in task_splits
            and row.get("split") == task_splits.get(row.get("task_id"))
            for row in raw_trajectories
        ),
        "scores_binary_finite": all(
            is_binary_score(row.get("score")) for row in raw_trajectories
        ),
        "schema_verified": all(
            row.get("schema_verified") is True for row in raw_trajectories
        ),
        "thinking_disabled": tools["nonempty_thinking_events"] == 0,
        "audit_covers_trajectories": {
            row.get("trajectory_id")
            for row in audit_rows
            if row.get("event") == "generation"
        }
        == seen_trajectory_ids,
        "api_no_terminal_failure": bool(api_rows)
        and all(row.get("success") or row.get("retry") for row in api_rows),
        "metrics_step_zero_only": bool(metrics_rows)
        and all(row.get("step") == 0 for row in metrics_rows),
        "no_actor_or_gradient_metrics": all(
            not contains_training_metric(row) for row in metrics_rows
        ),
        "no_checkpoint_files": not checkpoints,
    }
    job_start = read_epoch(run / "job_started.epoch")
    job_finish = read_epoch(run / "job_finished.epoch")
    summary = {
        "schema_version": 2,
        "run": str(run),
        "accepted": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "exit_code": exit_code,
        "observed": {
            "trajectory_rows": len(raw_trajectories),
            "unique_trajectories": len(unique_trajectories),
            "duplicate_trajectory_ids": duplicate_ids,
            "metrics_rows": len(metrics_rows),
        },
        "splits": {
            "train": aggregate_split(task_rows, "train"),
            "dev": aggregate_split(task_rows, "dev"),
        },
        "regression_tasks": {
            str(row["task_id"]): row
            for row in task_rows
            if row["task_id"] in REGRESSION_TASK_IDS
        },
        "tools": tools,
        "termination_reasons": dict(
            sorted(
                Counter(
                    str(row.get("termination", "missing"))
                    for row in unique_trajectories
                ).items()
            )
        ),
        "trajectories": {
            "latency_s": {
                "mean": mean(latencies),
                "p50": percentile(latencies, 0.5),
                "p90": percentile(latencies, 0.9),
                "p99": percentile(latencies, 0.99),
            },
            "assistant_tokens": {
                "total": sum(assistant_tokens),
                "mean": mean(assistant_tokens),
                "p50": percentile(assistant_tokens, 0.5),
                "p90": percentile(assistant_tokens, 0.9),
            },
            "tool_calls_total": sum(
                row.get("tool_calls", 0)
                for row in unique_trajectories
                if isinstance(row.get("tool_calls", 0), int)
            ),
        },
        "api": summarize_api(api_rows),
        "wall_time_s": job_finish - job_start
        if job_start is not None and job_finish is not None
        else None,
        "checkpoint_files": checkpoints,
    }
    return summary, task_rows


def format_value(value: Any) -> str:
    return (
        "未取得"
        if value is None
        else f"{value:.4f}"
        if isinstance(value, float)
        else str(value)
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# E00 原始 Qwen3-8B 能力评估报告",
        "",
        f"验收：{'通过' if summary['accepted'] else '未通过或尚未完成'}；退出码：{summary['exit_code']}。",
        "",
        "## Train 与 Dev 结果",
        "",
        r"| 集合 | 任务数 | 轨迹数 | 成功数 | $\mathrm{pass}^{1}$（任务成功率） | "
        r"$\mathrm{pass}^{4}$（四次全部成功） | $\mathrm{pass@4}$（四次至少一次成功） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "dev"):
        row = summary["splits"][name]
        lines.append(
            f"| {name} | {row['tasks']} | {row['n']} | {row['c']} | {format_value(row['success_rate'])} | "
            f"{format_value(row['pass_all_4'])} | {format_value(row['pass_at_4'])} |"
        )
    lines += [
        "",
        "各指标按任务等权平均；任务成功率即宏平均成功率。test 未参与本轮评测。",
        "",
        "## 回归任务",
        "",
    ]
    for task_id, row in summary["regression_tasks"].items():
        lines.append(
            f"- 任务 {task_id}：{row['c']}/{row['n']}，成功率 {format_value(row['success_rate'])}。"
        )
    tools = summary["tools"]
    lines += [
        "",
        "## 工具、轨迹与 API",
        "",
        (
            f"工具尝试 {tools['attempts']} 次，执行 {tools['executions']} 次，"
            f"严格合法率 {format_value(tools['strict_valid_rate'])}，错误响应 {tools['error_responses']} 次。"
        ),
        f"终止原因：{summary['termination_reasons']}。",
        f"轨迹延迟：{summary['trajectories']['latency_s']}；assistant token：{summary['trajectories']['assistant_tokens']}。",
        f"API：{summary['api']}。",
        f"作业总耗时：{format_value(summary['wall_time_s'])} 秒。",
        "",
        "## 验收明细",
        "",
    ]
    lines.extend(f"- {name}：{value}" for name, value in summary["checks"].items())
    if summary["errors"]:
        lines += ["", f"缺失或损坏的输入：{summary['errors']}。"]
    lines += ["", "未完成运行只汇报已观察数据；缺失轨迹不会按失败计入指标。", ""]
    return "\n".join(lines)


def write_outputs(
    run: Path, summary: dict[str, Any], task_rows: list[dict[str, Any]]
) -> None:
    (run / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run / "task_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    (run / "baseline_report.md").write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary, task_rows = build_summary(args.run)
    write_outputs(args.run, summary, task_rows)
    print(
        f"E00_SUMMARY_READY accepted={summary['accepted']} path={args.run / 'baseline_summary.json'}",
        flush=True,
    )
    raise SystemExit(0 if summary["accepted"] else 1)


if __name__ == "__main__":
    main()
