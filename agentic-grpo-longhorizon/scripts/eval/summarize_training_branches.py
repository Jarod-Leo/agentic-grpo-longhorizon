#!/usr/bin/env python3
"""Validate experiment-C branch records and write its signal summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.local_branching import (
    read_training_branch_records,
    summarize_training_branches,
)
from src.evaluation.run_metrics import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=2)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()

    roots, errors = read_jsonl(args.run_dir / "eval_trajectories.jsonl")
    if errors:
        raise ValueError(f"Invalid root trajectories: {errors}")
    records = read_training_branch_records(args.run_dir)
    summary = summarize_training_branches(
        records,
        roots,
        candidates=args.candidates,
        continuations=args.continuations,
    )
    accepted = summary["engineering_accepted"]
    signal = summary["nonzero_signal_groups"] > 0
    stable = (
        args.continuations < 4
        or summary["split_half_stable_signal_groups"] > 0
    )
    summary["signal_gate_passed"] = accepted and signal and stable
    summary["stopped_reason"] = (
        None
        if summary["signal_gate_passed"]
        else (
            "branch_engineering_checks_failed"
            if not accepted
            else (
                "zero_local_signal"
                if not signal
                else "unstable_split_half_signal"
            )
        )
    )
    path = args.run_dir / "local_branch_training_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(args.require_signal and not summary["signal_gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
