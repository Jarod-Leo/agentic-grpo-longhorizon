"""Prepare the exact fixed-budget task rows for verified SFT collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/train/grpo"))
from build_grpo_parquet import build_rows  # noqa: E402


SMOKE_TASK_IDS = [0, 1, 4, 6, 13, 45]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", choices=["teacher", "correction"], required=True)
    parser.add_argument("--task-ids", default=",".join(map(str, SMOKE_TASK_IDS)))
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()

    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    if len(task_ids) != len(set(task_ids)) or args.samples < 1:
        raise ValueError("Task IDs must be unique and samples must be positive")
    split_path = PROJECT_ROOT / "experiments/sft_collect_airline/split.json"
    seen_ids = {int(value) for value in json.loads(split_path.read_text())["seen_task_ids"]}
    outside = sorted(set(task_ids) - seen_ids)
    if outside:
        raise ValueError(f"Collection is restricted to the 40 train tasks: {outside}")

    pd.DataFrame(build_rows(task_ids, "train")).to_parquet(args.run / "eval.parquet", index=False)
    metadata_path = args.run / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["eval_split"] = "train"
    metadata["train_task_ids"] = task_ids
    metadata["test_task_ids"] = []
    metadata["collection_task_ids"] = task_ids
    metadata["samples_per_task"] = args.samples
    metadata["expected_trajectories"] = len(task_ids) * args.samples
    metadata["verified_sft_collection"] = {
        "mode": args.mode,
        "task_ids": task_ids,
        "attempts_per_task": args.samples,
        "fixed_budget": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"VERIFIED_COLLECTION_READY mode={args.mode} tasks={len(task_ids)} "
        f"attempts={len(task_ids) * args.samples}"
    )


if __name__ == "__main__":
    main()
