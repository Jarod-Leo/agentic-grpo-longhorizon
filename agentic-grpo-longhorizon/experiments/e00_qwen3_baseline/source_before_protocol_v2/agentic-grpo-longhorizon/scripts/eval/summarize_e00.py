"""Validate and summarize an E00 Qwen3-8B baseline evaluation run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pass_metrics import task_pass_metrics  # noqa: E402

EXPECTED_SAMPLES_PER_TASK = 8
REGRESSION_TASK_IDS = (11, 18, 28, 40)


def read_json(path: Path, default: Any) -> tuple[Any, list[str]]:
    if not path.exists():
        return default, [f"missing:{path.name}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as error:
        return default, [f"invalid:{path.name}:{error}"]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"missing:{path.name}"]
    rows = []
    errors = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append(f"invalid:{path.name}:{line_number}:not_an_object")
        except json.JSONDecodeError as error:
            errors.append(f"invalid:{path.name}:{line_number}:{error.msg}")
    return rows, errors


def mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.fmean(available) if available else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def as_timestamp(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_epoch(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        value = float(path.read_text(encoding="utf-8").strip())
        return value if math.isfinite(value) else None
    except (OSError, ValueError):
        return None


def is_binary_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value in (0, 1)
    )


def metric_value(mapping: dict[Any, Any], key: int) -> float | None:
    return mapping.get(key, mapping.get(str(key)))


def per_task_results(
    trajectories: list[dict[str, Any]],
    task_splits: dict[int, str],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        task_id = row.get("task_id")
        if (
            task_id in task_splits
            and row.get("split") == task_splits[task_id]
            and is_binary_score(row.get("score"))
        ):
            grouped[task_id].append(row)

    results = []
    for task_id in sorted(task_splits):
        rows = grouped[task_id]
        n = len(rows)
        c = sum(int(row["score"]) for row in rows)
        metrics = task_pass_metrics(n, c, ks=(1, 4, 8))
        results.append(
            {
                "task_id": task_id,
                "split": task_splits[task_id],
                "n": n,
                "c": c,
                "success_rate": metrics["success_rate"],
                "pass_at_k": {
                    str(k): value for k, value in metrics["pass_at_k"].items()
                },
                "pass_all_k": {
                    str(k): value for k, value in metrics["pass_all_k"].items()
                },
            }
        )
    return results


def aggregate_split(task_rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    rows = [row for row in task_rows if row["split"] == split_name]
    return {
        "tasks": len(rows),
        "n": sum(row["n"] for row in rows),
        "c": sum(row["c"] for row in rows),
        "success_rate": mean([row["success_rate"] for row in rows]),
        "pass_at_4": mean([metric_value(row["pass_at_k"], 4) for row in rows]),
        "pass_all_4": mean([metric_value(row["pass_all_k"], 4) for row in rows]),
    }


def summarize_tools(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    generations = [row for row in audit_rows if row.get("event") == "generation"]
    executions = [row for row in audit_rows if row.get("event") == "execution"]
    attempts = sum(row.get("tool_tag_starts", 0) for row in generations)
    strict_valid = sum(
        call.get("json_valid") is True
        and call.get("known_name") is True
        and call.get("schema_valid") is True
        for row in generations
        for call in row.get("calls", [])
    )
    return {
        "generation_events": len(generations),
        "attempts": attempts,
        "executions": len(executions),
        "strict_valid": strict_valid,
        "strict_valid_rate": strict_valid / attempts if attempts else None,
        "error_responses": sum(row.get("error") is True for row in executions),
        "nonempty_thinking_events": sum(
            row.get("nonempty_thinking") is True for row in generations
        ),
    }


def summarize_api(api_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (row for row in api_rows if as_timestamp(row.get("started_at")) is not None),
        key=lambda row: as_timestamp(row["started_at"]),
    )
    usage_names = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {
        name: sum((row.get("usage") or {}).get(name, 0) or 0 for row in api_rows)
        for name in usage_names
    }
    latencies = [
        float(row["latency_s"])
        for row in api_rows
        if isinstance(row.get("latency_s"), (int, float))
    ]
    window: deque[dict[str, Any]] = deque()
    peak_rpm = 0
    peak_tpm = 0
    for row in ordered:
        started = as_timestamp(row["started_at"])
        window.append(row)
        while window and as_timestamp(window[0]["started_at"]) <= started - 60:
            window.popleft()
        peak_rpm = max(peak_rpm, len(window))
        peak_tpm = max(
            peak_tpm,
            sum(
                (item.get("usage") or {}).get("total_tokens", 0) or 0 for item in window
            ),
        )
    if ordered:
        first = as_timestamp(ordered[0]["started_at"])
        finishes = [as_timestamp(row.get("finished_at")) for row in ordered]
        valid_finishes = [value for value in finishes if value is not None]
        span = max(valid_finishes) - first if valid_finishes else None
    else:
        span = None
    return {
        "requests": len(api_rows),
        "successful_requests": sum(row.get("success") is True for row in api_rows),
        "failed_attempts": sum(row.get("success") is False for row in api_rows),
        "retry_attempts": sum((row.get("attempt") or 1) > 1 for row in api_rows),
        "usage": usage,
        "latency_s": {
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "p99": percentile(latencies, 0.99),
        },
        "active_span_s": span,
        "mean_rpm": len(api_rows) * 60 / span if span and span > 0 else None,
        "peak_rolling_60s_rpm": peak_rpm,
        "peak_rolling_60s_tpm": peak_tpm,
    }


def contains_training_metric(row: dict[str, Any]) -> bool:
    data = row.get("data", row)
    if not isinstance(data, dict):
        return False
    return any(str(key).startswith("actor/") or "grad_norm" in str(key) for key in data)


def checkpoint_files(run: Path) -> list[str]:
    found = []
    for path in run.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if any("checkpoint" in part.lower() for part in relative.parts):
            found.append(str(relative))
    return sorted(found)


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
