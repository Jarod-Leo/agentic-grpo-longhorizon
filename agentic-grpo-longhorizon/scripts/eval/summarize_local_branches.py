"""Validate and summarize an experiment-C local branch diagnostic run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.local_branch_metrics import build_local_branch_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    summary = build_local_branch_summary(args.run_dir)
    output = args.run_dir / "branch_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["engineering_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
