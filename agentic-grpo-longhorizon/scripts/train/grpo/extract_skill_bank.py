"""Collect train successes or extract a fixed skill bank inside a CPU Slurm job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.training.skill_bank import (
    collect_successful_trajectories,
    extract_skill_bank,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser(
        "collect", help="No model/API calls; retain only accepted train successes"
    )
    collect.add_argument("--source-run", type=Path, required=True)
    collect.add_argument("--split", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--max-traces-per-task", type=int, default=4)
    extract = commands.add_parser(
        "extract", help="Real MiMo API calls with resumable responses"
    )
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run data processing and extraction inside Slurm")
    if args.command == "collect":
        result = collect_successful_trajectories(
            args.source_run, args.split, args.output_dir, args.max_traces_per_task
        )
    else:
        from src.envs.mimo_client import MimoClient
        from transformers import AutoTokenizer

        args.output.parent.mkdir(parents=True, exist_ok=True)
        os.environ["MIMO_METRICS_PATH"] = str(args.output.with_suffix(".api.jsonl"))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        client = MimoClient(
            {
                "user_model": "mimo-v2.5",
                "rpm": 100,
                "tpm": 10_000_000,
                "max_inflight": 1,
                "max_completion_tokens": 1024,
            }
        )
        bank = extract_skill_bank(args.manifest, client, tokenizer, args.output)
        result = {
            "requested_candidates": bank["requested_candidates"],
            "retained_candidates": bank["retained_candidates"],
            "covered_tasks": len({skill["task_id"] for skill in bank["skills"]}),
            "bank_path": str(args.output),
            "bank_sha256": file_sha256(args.output),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
