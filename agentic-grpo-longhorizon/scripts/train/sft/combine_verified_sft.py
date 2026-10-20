"""Combine independently gated teacher and correction parquet files for B SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=4)
    args = parser.parse_args()

    teacher = pd.read_parquet(args.teacher)
    correction = pd.read_parquet(args.correction)
    combined = pd.concat([teacher, correction], ignore_index=True)
    reasons = []
    if teacher.empty or not teacher["is_write_task"].astype(bool).any():
        reasons.append("teacher data has no verified write-task success")
    if correction.empty:
        reasons.append("correction data has no verified repair success")
    if len(combined) < args.min_samples:
        reasons.append(f"at least {args.min_samples} training samples are required, found {len(combined)}")
    if combined["trajectory_id"].duplicated().any():
        reasons.append("duplicate trajectory IDs across teacher/correction data")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "teacher": len(teacher),
        "correction": len(correction),
        "total": len(combined),
        "tasks": sorted({int(task_id) for task_id in combined["task_id"]}),
        "output": str(args.output),
        "status": "rejected" if reasons else "accepted",
        "reasons": reasons,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if reasons:
        raise RuntimeError("; ".join(reasons))

    combined.to_parquet(args.output, index=False)
    print(
        f"VERIFIED_SFT_COMBINED teacher={len(teacher)} correction={len(correction)} "
        f"total={len(combined)}"
    )


if __name__ == "__main__":
    main()
