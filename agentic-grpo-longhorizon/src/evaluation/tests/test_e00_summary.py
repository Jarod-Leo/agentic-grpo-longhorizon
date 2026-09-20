from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "e00_summary", HERE / "scripts/eval/summarize_e00.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def create_complete_run(run: Path) -> None:
    checked_split = json.loads(
        (HERE / "experiments/e00_qwen3_baseline/split.json").read_text(encoding="utf-8")
    )
    (run / "split.json").write_text(json.dumps(checked_split), encoding="utf-8")
    task_splits = {task_id: "train" for task_id in checked_split["train_task_ids"]}
    task_splits.update({task_id: "dev" for task_id in checked_split["dev_task_ids"]})
    trajectories = []
    audit = []
    for task_id, split_name in task_splits.items():
        for sample in range(8):
            trajectory_id = f"{task_id}-{sample}"
            trajectories.append(
                {
                    "task_id": task_id,
                    "split": split_name,
                    "trajectory_id": trajectory_id,
                    "score": int(sample < task_id % 9),
                    "termination": "done",
                    "assistant_tokens": 10 + sample,
                    "tool_calls": 1,
                    "started_at": 1000 + sample,
                    "finished_at": 1001 + sample,
                    "schema_verified": True,
                    "messages": [],
                    "rendered_transcript": "",
                }
            )
            audit.extend(
                [
                    {
                        "event": "generation",
                        "trajectory_id": trajectory_id,
                        "tool_tag_starts": 1,
                        "calls": [
                            {
                                "json_valid": True,
                                "known_name": True,
                                "schema_valid": True,
                            }
                        ],
                        "nonempty_thinking": False,
                    },
                    {
                        "event": "execution",
                        "trajectory_id": trajectory_id,
                        "error": False,
                        "response": "ok",
                    },
                ]
            )
    write_jsonl(run / "eval_trajectories.jsonl", trajectories)
    write_jsonl(run / "tool_audit.jsonl", audit)
    write_jsonl(
        run / "api.jsonl",
        [
            {
                "started_at": 1000,
                "finished_at": 1001,
                "latency_s": 1,
                "attempt": 1,
                "success": True,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ],
    )
    write_jsonl(run / "metrics.jsonl", [{"step": 0, "data": {"val/success": 0.5}}])
    (run / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run / "job_started.epoch").write_text("1000\n", encoding="utf-8")
    (run / "job_finished.epoch").write_text("1100\n", encoding="utf-8")


def test_complete_run_is_accepted_and_written(tmp_path):
    create_complete_run(tmp_path)
    module = load_module()
    summary, task_rows = module.build_summary(tmp_path)

    assert summary["accepted"] is True
    assert summary["observed"]["unique_trajectories"] == 320
    assert summary["splits"]["train"]["n"] == 240
    assert summary["splits"]["dev"]["n"] == 80
    assert summary["tools"]["strict_valid_rate"] == 1.0
    assert summary["wall_time_s"] == 100
    assert len(task_rows) == 40

    module.write_outputs(tmp_path, summary, task_rows)
    assert (
        json.loads((tmp_path / "baseline_summary.json").read_text())["accepted"] is True
    )
    assert len((tmp_path / "task_results.jsonl").read_text().splitlines()) == 40
    assert "E00 原始 Qwen3-8B" in (tmp_path / "baseline_report.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "test", "step_one"])
def test_invalid_run_is_rejected_without_counting_missing_as_failure(
    tmp_path, mutation
):
    create_complete_run(tmp_path)
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "eval_trajectories.jsonl").read_text().splitlines()
    ]
    if mutation == "missing":
        trajectories.pop()
        write_jsonl(tmp_path / "eval_trajectories.jsonl", trajectories)
    elif mutation == "duplicate":
        trajectories[-1] = dict(trajectories[0])
        write_jsonl(tmp_path / "eval_trajectories.jsonl", trajectories)
    elif mutation == "test":
        trajectories[-1]["task_id"] = 3
        trajectories[-1]["split"] = "test"
        write_jsonl(tmp_path / "eval_trajectories.jsonl", trajectories)
    else:
        write_jsonl(
            tmp_path / "metrics.jsonl", [{"step": 1, "data": {"actor/grad_norm": 1.0}}]
        )

    module = load_module()
    summary, task_rows = module.build_summary(tmp_path)

    assert summary["accepted"] is False
    if mutation == "missing":
        affected = next(
            row for row in task_rows if row["task_id"] == trajectories[-1]["task_id"]
        )
        assert affected["n"] == 7
        assert affected["success_rate"] == affected["c"] / 7
