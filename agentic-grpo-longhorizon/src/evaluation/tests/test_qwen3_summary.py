"""Completeness must follow task IDs and repeats, including zero-reward results."""

import json
import runpy
from pathlib import Path

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
