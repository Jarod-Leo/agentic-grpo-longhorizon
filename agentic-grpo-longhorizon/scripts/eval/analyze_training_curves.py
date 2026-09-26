"""Analyze shared Qwen3 run logs offline; run through Slurm on the cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.run_metrics import aggregate_split, is_binary_score, per_task_results


TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TRAJECTORY_FIELDS = (
    "trajectory_id", "step", "validate", "task_id", "split", "score", "termination",
    "assistant_tokens", "assistant_turns", "tool_calls",
)


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream input so transcripts and tool responses never accumulate in memory."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if finite(value):
                values[key].append(value)
    return {
        key: {"n": len(items), "mean": mean(items), "min": min(items), "max": max(items)}
        for key, items in sorted(values.items())
    }


def repeated_calls(messages: Any) -> tuple[int | None, int]:
    """Count exact repeated tool names/JSON arguments, not merely repeated tool names."""
    if not isinstance(messages, list):
        return None, 0
    signatures: list[str] = []
    malformed = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({"name": function.get("name"), "arguments": arguments})
            except json.JSONDecodeError:
                malformed += 1
        content = message.get("content")
        if not calls and isinstance(content, str):
            for block in TOOL_BLOCK.findall(content):
                try:
                    calls.append(json.loads(block))
                except json.JSONDecodeError:
                    malformed += 1
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                malformed += 1
                continue
            if "arguments" not in call:
                malformed += 1
                continue
            signatures.append(json.dumps(
                [call["name"], call["arguments"]], sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ))
    return len(signatures) - len(set(signatures)), malformed


def task_splits(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[int, str]:
    splits = {
        task: split
        for split in ("train", "test")
        for task in metadata.get(f"{split}_task_ids", [])
    }
    return splits or {row["task_id"]: row["split"] for row in rows}


def trajectory_summary(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if is_binary_score(row.get("score"))]
    per_task = per_task_results(valid, task_splits(valid, metadata))
    sampled_tasks = [task for task in per_task if task["n"]]
    limited = [row for row in rows if row.get("termination") == "assistant_turn_limit"]
    repeat_known = [row for row in rows if row["repeated_calls"] is not None]
    split_names = sorted({row.get("split") for row in valid})
    aggregates = {split: aggregate_split(per_task, split) for split in split_names}
    for split in aggregates:
        split_tasks = [task for task in sampled_tasks if task["split"] == split]
        for key in ("pass_at_k", "pass_all_k"):
            aggregates[split][key] = {
                str(k): mean([task[key][str(k)] for task in split_tasks if task[key][str(k)] is not None])
                for k in (1, 4, 8)
            }
    return {
        "trajectories": len(rows), "binary_score_trajectories": len(valid),
        "successful_trajectories": sum(int(row["score"]) for row in valid),
        "sample_success_rate": mean([row["score"] for row in valid]),
        "means": {
            key: mean([row[key] for row in rows if finite(row.get(key))])
            for key in ("assistant_tokens", "assistant_turns", "tool_calls")
        },
        "termination_counts": dict(sorted(Counter(row.get("termination") for row in rows).items())),
        "assistant_turn_limit_trajectories": len(limited),
        "assistant_turn_limit_rate": len(limited) / len(rows) if rows else None,
        "assistant_turn_limit_failures": sum(row.get("score") == 0 for row in limited),
        "repeat_calls": {
            "assessable_trajectories": len(repeat_known),
            "trajectories_with_exact_repeats": sum(row["repeated_calls"] > 0 for row in repeat_known),
            "trajectory_fraction": mean([int(row["repeated_calls"] > 0) for row in repeat_known]),
            "extra_repeated_calls": sum(row["repeated_calls"] for row in repeat_known),
            "unparsed_calls": sum(row["unparsed_calls"] for row in rows),
        },
        "task_success_count_distribution": dict(sorted(Counter(task["c"] for task in sampled_tasks).items())),
        "per_task": per_task, "splits": aggregates,
    }


def group_key(row: dict[str, Any]) -> str:
    step = int(row.get("step", 0))
    if row.get("validate"):
        return f"eval_step{step}"
    start = ((step - 1) // 50) * 50 + 1
    return f"train_steps{start}-{start + 49}"


def load_run(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metrics = sorted(jsonl(run_dir / "metrics.jsonl"), key=lambda row: row["step"])
    trajectory_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    trajectory_to_group = {}
    for row in jsonl(run_dir / "eval_trajectories.jsonl"):
        repeats, malformed = repeated_calls(row.get("messages"))
        compact = {key: row.get(key) for key in TRAJECTORY_FIELDS}
        if metadata.get("mode") == "eval":
            compact["validate"] = True
        compact.update(repeated_calls=repeats, unparsed_calls=malformed)
        key = group_key(compact)
        trajectory_groups[key].append(compact)
        trajectory_to_group[compact["trajectory_id"]] = key
        if not compact["validate"]:
            train_steps[int(compact["step"])].append(compact)
    audit_groups: dict[str, Counter] = defaultdict(Counter)
    audit_error_names: dict[str, Counter] = defaultdict(Counter)
    unmatched = 0
    for row in jsonl(run_dir / "tool_audit.jsonl"):
        key = trajectory_to_group.get(row.get("trajectory_id"))
        if key is None:
            unmatched += 1
            continue
        counts = audit_groups[key]
        if row.get("event") == "execution":
            counts["executions"] += 1
            if row.get("error") is True:
                counts["error_responses"] += 1
                audit_error_names[key][row.get("name", "unknown")] += 1
        elif row.get("event") == "generation":
            counts["generation_events"] += 1
            counts["nonempty_thinking_events"] += row.get("nonempty_thinking") is True
    groups = {}
    for key, rows in sorted(trajectory_groups.items()):
        group = trajectory_summary(rows, metadata)
        counts = dict(audit_groups[key])
        executions = counts.get("executions", 0)
        counts["error_response_rate"] = counts.get("error_responses", 0) / executions if executions else None
        counts["error_tool_counts"] = dict(audit_error_names[key].most_common())
        group["tool_audit"] = counts
        groups[key] = group
    blocks = []
    for start in range(1, max((int(row["step"]) for row in metrics), default=0) + 1, 50):
        selected = [row for row in metrics if start <= int(row["step"]) <= start + 49]
        blocks.append({
            "start_step": start, "end_step": start + 49, "observed_steps": len(selected),
            "metrics": numeric_summary([row.get("data", row) for row in selected]),
        })
    return {
        "run_dir": str(run_dir.resolve()), "metadata": metadata,
        "summary": summary, "per_step_metrics": metrics, "metric_blocks": blocks,
        "trajectory_groups": groups,
        "per_step_trajectories": {
            str(step): trajectory_summary(rows, {}) for step, rows in sorted(train_steps.items())
        },
        "unmatched_tool_audit_events": unmatched,
    }


def moving_average(values: list[float], window: int = 10) -> list[float]:
    return [math.nan if index < window - 1 else statistics.fmean(values[index - window + 1:index + 1])
            for index in range(len(values))]


def plot_curves(analysis: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 150})
    run = analysis["run"]
    metrics = run["per_step_metrics"]

    def series(key: str) -> tuple[list[int], list[float]]:
        rows = [(int(row["step"]), row.get("data", row).get(key)) for row in metrics]
        return [step for step, value in rows if finite(value)], [value for _, value in rows if finite(value)]

    def draw(axis: Any, steps: list[int], values: list[float], label: str, color: str) -> None:
        axis.plot(steps, values, color=color, alpha=0.25, linewidth=0.8, label=f"{label}: raw")
        axis.plot(steps, moving_average(values), color=color, linewidth=1.8, label=f"{label}: trailing 10 steps")

    def finish(fig: Any, axis: Any, name: str, title: str, ylabel: str, note: str = "", legend_loc: str = "best") -> None:
        axis.set(title=title, xlabel="Training step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc=legend_loc, fontsize=8)
        if note:
            fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.045 if note else 0, 1, 1))
        fig.savefig(output_dir / f"{name}.png")
        fig.savefig(output_dir / f"{name}.pdf")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    steps, values = series("critic/score/mean")
    draw(axis, steps, values, "Train rollout success", "tab:blue")
    for key, group in run["trajectory_groups"].items():
        if key.startswith("eval_step"):
            step = int(key.removeprefix("eval_step"))
            value = group["splits"].get("train", {}).get("success_rate")
            if value is not None:
                axis.scatter([step], [value], color="tab:red", zorder=4, label=f"Train eval step {step}: {value:.3%}")
    baseline = analysis.get("baseline", {}).get("summary", {}).get("splits", {}).get("train", {}).get("success_rate")
    if finite(baseline):
        axis.axhline(baseline, color="black", linestyle="--", label=f"Base model train eval: {baseline:.2%}")
    finish(fig, axis, "01_task_success", "Task success", "Success rate", "Train rollouts and fixed-task evaluations use different sampling temperatures.")

    specs = (
        ("actor/opd_loss", "02_opd_loss", "Logged OPD loss", "Loss", "Microbatch-scaled/aggregated training surrogate; not exact vocabulary KL."),
        ("actor/opd_advantage_mean", "03_teacher_student_gap", "Negative logged OPD advantage", "Negative advantage", "Logged microbatch statistic; not a calibrated estimate of exact KL."),
        ("actor/grad_norm", "04_gradient_norm", "Actor gradient norm", "Gradient norm", ""),
        ("actor/opd_clipfrac", "06_opd_clip_fraction", "OPD clipping fraction", "Fraction", "On-policy sampled-token loss; clipping is not a teacher-quality metric."),
    )
    for key, name, title, ylabel, note in specs:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        steps, values = series(key)
        if key == "actor/opd_advantage_mean":
            values = [-value for value in values]
        draw(axis, steps, values, key, "tab:blue")
        finish(fig, axis, name, title, ylabel, note)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    trajectories = run["per_step_trajectories"]
    steps = [int(step) for step in trajectories]
    tokens = [group["means"]["assistant_tokens"] for group in trajectories.values()]
    limits = [group["assistant_turn_limit_rate"] for group in trajectories.values()]
    draw(axis, steps, tokens, "Assistant tokens", "tab:blue")
    right = axis.twinx()
    draw(right, steps, limits, "Assistant turn limit", "tab:orange")
    right.set_ylabel("Turn limit rate")
    right.set_ylim(0, 1)
    right.legend(loc="upper right", fontsize=8)
    finish(fig, axis, "05_trajectory_length", "Trajectory length and turn limit", "Mean assistant tokens", legend_loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path)
    args = parser.parse_args()
    analysis = {
        "schema_version": 1,
        "notes": [
            "All logged metric values are preserved; block statistics weight training steps equally.",
            "Logged OPD loss is microbatch-scaled and aggregated; neither this loss nor negative advantage is exact KL.",
            "Pass metrics use shared tau combinatorial estimators and macro-average task metrics.",
            "A repeated tool call means identical name and JSON arguments; repetition does not by itself prove an error.",
            "Turn limit rate counts assistant_turn_limit termination, not token truncation.",
        ],
        "run": load_run(args.run_dir),
    }
    if args.baseline_run_dir:
        analysis["baseline"] = load_run(args.baseline_run_dir)
    input_paths = [args.run_dir]
    if args.baseline_run_dir:
        input_paths.append(args.baseline_run_dir)
    analysis["input_sha256"] = {}
    for directory in input_paths:
        for name in ("metrics.jsonl", "eval_trajectories.jsonl", "tool_audit.jsonl", "run.json", "summary.json"):
            path = directory / name
            if path.exists():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                analysis["input_sha256"][str(path.resolve())] = digest.hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "analysis.json"
    destination.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    plot_curves(analysis, args.output_dir)
    print(json.dumps({"analysis": str(destination.resolve()), "training_steps": len(analysis["run"]["per_step_metrics"]), "plots": 6}))


if __name__ == "__main__":
    main()
