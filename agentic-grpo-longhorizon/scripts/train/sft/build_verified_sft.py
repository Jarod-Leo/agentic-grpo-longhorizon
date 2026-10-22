"""Filter successful, uncontaminated collector records into token-exact SFT parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


_WRITE_TASK_IDS = {0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 14, 19, 22, 25, 27, 28, 31, 32, 33, 34, 45}
_VALID_SUCCESS_TERMINATIONS = {"tool_terminated", "interaction_terminated", "environment_terminated"}


def select_verified_records(records: list[dict], seen_ids: set[int], mode: str):
    selected = []
    rejected = Counter()
    for record in records:
        reason = None
        if record.get("sft_mode") != mode:
            reason = "other_mode"
        elif int(record.get("task_id", -1)) not in seen_ids or record.get("split") not in {"train", "seen"}:
            reason = "outside_train_split"
        elif record.get("collection_failure"):
            reason = f"collection_failure:{record['collection_failure']}"
        elif float(record.get("score", 0.0)) != 1.0:
            reason = "environment_failure"
        elif record.get("terminal_semantics") != "first_environment_done_v1":
            reason = "wrong_terminal_semantics"
        elif record.get("termination") not in _VALID_SUCCESS_TERMINATIONS:
            reason = "invalid_success_termination"
        elif record.get("contaminated", False):
            reason = "contaminated"
        elif not record.get("teacher_takeover", False):
            reason = "no_teacher_takeover"
        else:
            input_ids = record.get("input_ids")
            loss_mask = record.get("loss_mask")
            if not isinstance(input_ids, list) or not isinstance(loss_mask, list):
                reason = "missing_token_data"
            elif len(input_ids) != len(loss_mask) or not any(loss_mask):
                reason = "invalid_teacher_mask"
        if reason:
            rejected[reason] += 1
            continue
        selected.append(
            {
                "input_ids": record["input_ids"],
                "loss_mask": record["loss_mask"],
                "task_id": int(record["task_id"]),
                "trajectory_id": record["trajectory_id"],
                "sft_mode": mode,
                "loss_start": int(record["loss_start"]),
                "is_write_task": int(record["task_id"]) in _WRITE_TASK_IDS,
            }
        )
    return selected, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--mode", choices=["teacher", "correction"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    seen_ids = {int(task_id) for task_id in split["seen_task_ids"]}
    if len(seen_ids) != 40:
        raise ValueError(f"Expected the fixed 40-task train split, got {len(seen_ids)}")

    records = []
    for path in args.input:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    selected, rejected = select_verified_records(records, seen_ids, args.mode)
    write_successes = sum(bool(row["is_write_task"]) for row in selected)
    summary = {
        "mode": args.mode,
        "attempts": len(records),
        "accepted": len(selected),
        "accepted_tasks": sorted({row["task_id"] for row in selected}),
        "write_successes": write_successes,
        "rejected": dict(sorted(rejected.items())),
        "output": str(args.output),
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if not selected:
        label = "repair" if args.mode == "correction" else "teacher"
        raise RuntimeError(f"No verified {label} success; stop before SFT. See {summary_path}")
    if args.mode == "teacher" and write_successes == 0:
        raise RuntimeError(f"No successful write-task trajectory; stop before SFT. See {summary_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_parquet(args.output, index=False)
    print(
        f"VERIFIED_SFT_READY mode={args.mode} accepted={len(selected)} "
        f"tasks={len(summary['accepted_tasks'])} write_successes={write_successes}"
    )


if __name__ == "__main__":
    main()
