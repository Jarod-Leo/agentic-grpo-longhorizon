#!/usr/bin/env python3
"""Backfill scalar metrics from a completed Qwen3 experiment to W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECEIPT_NAME = "wandb_upload.json"
CONFIG_FIELDS = (
    "protocol",
    "config_name",
    "reward_mode",
    "adv_estimator",
    "lata_alpha",
    "prm_coefficient",
    "mode",
    "eval_split",
    "samples_per_task",
    "train_batch_size",
    "total_steps",
    "start_step",
    "checkpoint_steps",
    "evaluation_steps",
    "eval_samples_per_task",
    "base_model",
    "seed",
    "split_sha256",
    "slurm_job_id",
)
EVAL_FIELDS = {
    "success_rate": "pass1",
    "pass_all_4": "pass_all_4",
    "pass_at_4": "pass_at_4",
    "tasks": "tasks",
    "n": "trajectories",
    "c": "successes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload an existing metrics.jsonl plus audited train/test evaluation summaries to W&B. "
            "Only scalar metrics and a fixed metadata whitelist are sent."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Directory containing metrics.jsonl, run.json, and summary.json")
    parser.add_argument("--entity", required=True, help="W&B entity (user or team)")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument("--name", required=True, help="W&B run display name")
    parser.add_argument(
        "--test-summary",
        type=Path,
        help="Optional accepted held-out test summary.json to add under eval/test/*",
    )
    parser.add_argument(
        "--test-step",
        type=int,
        default=200,
        help="Training step at which to attach --test-summary metrics (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the import without importing W&B, requiring credentials, or writing a receipt",
    )
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def finite_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def read_metrics(path: Path) -> list[tuple[int, dict[str, int | float]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"required file does not exist: {path}") from exc

    rows: list[tuple[int, dict[str, int | float]]] = []
    previous_step: int | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank line in {path} at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc}") from exc
        if not isinstance(row, dict) or set(row) != {"step", "data"}:
            raise ValueError(f"{path}:{line_number} must contain exactly 'step' and 'data'")

        step = row["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"{path}:{line_number} has invalid non-negative integer step {step!r}")
        if previous_step is not None and step <= previous_step:
            raise ValueError(f"steps must be strictly increasing: {previous_step} then {step}")

        data = row["data"]
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{line_number} data must be an object")
        clean_data: dict[str, int | float] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path}:{line_number} contains an invalid metric name")
            clean_data[key] = finite_number(value, f"{path}:{line_number} metric {key!r}")
        logged_step = clean_data.get("training/global_step")
        if logged_step is not None and logged_step != step:
            raise ValueError(
                f"{path}:{line_number} outer step {step} does not match training/global_step {logged_step}"
            )
        rows.append((step, clean_data))
        previous_step = step

    if not rows:
        raise ValueError(f"no metric rows found in {path}")
    return rows


def failed_checks(summary: dict[str, Any], label: str) -> list[str]:
    accepted = summary.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError(f"{label}.accepted must be boolean")
    checks = summary.get("checks")
    if not isinstance(checks, dict) or any(
        not isinstance(name, str) or not isinstance(value, bool) for name, value in checks.items()
    ):
        raise ValueError(f"{label}.checks must map names to booleans")
    failed = sorted(name for name, value in checks.items() if not value)
    if accepted == bool(failed):
        raise ValueError(f"{label}.accepted is inconsistent with failed checks: {failed}")
    return failed


def evaluation_metrics(record: Any, split: str, label: str) -> dict[str, int | float]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    metrics: dict[str, int | float] = {}
    for source_name, target_name in EVAL_FIELDS.items():
        if source_name not in record:
            raise ValueError(f"{label} is missing {source_name!r}")
        metrics[f"eval/{split}/{target_name}"] = finite_number(record[source_name], f"{label}.{source_name}")
    return metrics


def add_metrics(target: dict[str, int | float], additions: dict[str, int | float], step: int) -> None:
    conflicts = sorted(set(target).intersection(additions))
    if conflicts:
        raise ValueError(f"step {step} already contains generated metric(s): {conflicts}")
    target.update(additions)


def merge_audited_evaluations(
    rows: list[tuple[int, dict[str, int | float]]],
    run_summary: dict[str, Any],
    test_summary: dict[str, Any] | None,
    test_step: int,
) -> None:
    rows_by_step = dict(rows)
    metadata = run_summary.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("summary.metadata must be an object")
    split = metadata.get("eval_split")
    if not isinstance(split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError(f"summary.metadata.eval_split is invalid: {split!r}")
    evaluations = run_summary.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError("summary.evaluations must be an object")
    for raw_step, record in evaluations.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid evaluation step {raw_step!r}") from exc
        if str(step) != str(raw_step) or step not in rows_by_step:
            raise ValueError(f"evaluation step {raw_step!r} has no matching metrics row")
        add_metrics(rows_by_step[step], evaluation_metrics(record, split, f"summary.evaluations[{raw_step!r}]"), step)

    if test_summary is None:
        return
    if test_summary.get("accepted") is not True:
        raise ValueError("--test-summary must have accepted=true")
    test_metadata = test_summary.get("metadata")
    if not isinstance(test_metadata, dict) or test_metadata.get("eval_split") != "test":
        raise ValueError("--test-summary metadata.eval_split must be 'test'")
    splits = test_summary.get("splits")
    if not isinstance(splits, dict) or "test" not in splits:
        raise ValueError("--test-summary must contain splits.test")
    if test_step not in rows_by_step:
        raise ValueError(f"--test-step {test_step} has no matching metrics row")
    add_metrics(
        rows_by_step[test_step],
        evaluation_metrics(splits["test"], "test", "test_summary.splits.test"),
        test_step,
    )


def make_config(run_metadata: dict[str, Any], accepted: bool, failed: list[str], has_test: bool) -> dict[str, Any]:
    config = {key: run_metadata[key] for key in CONFIG_FIELDS if key in run_metadata}
    config.update(
        {
            "audit_accepted": accepted,
            "audit_failed_checks": ",".join(failed) if failed else "none",
            "test_summary_included": has_test,
            "raw_val_metrics_split": run_metadata.get("eval_split"),
            "raw_val_metrics_note": (
                "Original val-* fields are preserved verbatim and describe the run metadata eval_split; "
                "use eval/<split>/* for audited pass metrics."
            ),
            "metrics_backfilled": True,
            "system_metrics_from_backfill_host": False,
        }
    )
    return config


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def make_run_id(entity: str, project: str, run_dir: Path) -> str:
    source = f"{entity}\0{project}\0{run_dir.resolve()}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()[:16]


def matching_receipt(receipt_path: Path, expected: dict[str, Any]) -> bool:
    if not receipt_path.exists():
        return False
    receipt = read_object(receipt_path)
    keys = ("upload_completed", "entity", "project", "name", "run_id", "metrics_sha256", "audit_sha256")
    if all(receipt.get(key) == expected.get(key) for key in keys):
        return True
    raise ValueError(
        f"{receipt_path} exists but does not match this upload target/content; refusing to overwrite it"
    )


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def upload(
    args: argparse.Namespace,
    rows: list[tuple[int, dict[str, int | float]]],
    config: dict[str, Any],
    accepted: bool,
    failed: list[str],
    test_accepted: bool | None,
    run_id: str,
) -> tuple[str, str]:
    if not os.environ.get("WANDB_API_KEY"):
        typo_hint = " (WANBD_API_KEY is misspelled; rename it)" if os.environ.get("WANBD_API_KEY") else ""
        raise ValueError(f"WANDB_API_KEY is not set{typo_hint}")
    try:
        import wandb
    except ImportError as exc:
        raise ValueError("the wandb SDK is unavailable; activate the project environment first") from exc

    tags = ["metrics-backfill", "audit-accepted" if accepted else "audit-failed"]
    if "thinking_disabled" in failed:
        tags.append("thinking-disabled-failed")
    settings = wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        disable_job_creation=True,
        x_disable_stats=True,
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        id=run_id,
        resume="never",
        mode="online",
        config=config,
        tags=tags,
        dir=str(args.run_dir.resolve()),
        save_code=False,
        settings=settings,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    try:
        for step, data in rows:
            run.log(data, step=step)
        audit_summary: dict[str, Any] = {
            "audit/accepted": accepted,
            "audit/failed_checks": ",".join(failed) if failed else "none",
            "audit/upload_succeeded": True,
        }
        if test_accepted is not None:
            audit_summary["audit/test_accepted"] = test_accepted
        run.summary.update(audit_summary)
        url = run.url or ""
        actual_id = run.id
        run.finish(exit_code=0 if accepted else 1)
    except Exception:
        run.finish(exit_code=1)
        raise
    return actual_id, url


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    run_path = run_dir / "run.json"
    summary_path = run_dir / "summary.json"
    test_path = args.test_summary.resolve() if args.test_summary else None

    run_metadata = read_object(run_path)
    run_summary = read_object(summary_path)
    rows = read_metrics(metrics_path)
    failed = failed_checks(run_summary, "summary")
    accepted = run_summary["accepted"]
    test_summary = read_object(test_path) if test_path else None
    test_failed = failed_checks(test_summary, "test_summary") if test_summary else []
    if test_summary is not None and test_failed:
        raise ValueError(f"--test-summary has failed checks: {test_failed}")
    merge_audited_evaluations(rows, run_summary, test_summary, args.test_step)

    summary_metadata = run_summary.get("metadata")
    if run_metadata != summary_metadata:
        raise ValueError("run.json does not exactly match summary.metadata")
    config = make_config(run_metadata, accepted, failed, test_summary is not None)
    run_id = make_run_id(args.entity, args.project, run_dir)
    source_paths = [metrics_path, run_path, summary_path]
    if test_path is not None:
        source_paths.append(test_path)
    receipt_base = {
        "upload_completed": True,
        "entity": args.entity,
        "project": args.project,
        "name": args.name,
        "run_id": run_id,
        "metrics_sha256": file_hash(metrics_path),
        "audit_sha256": audit_hash(source_paths),
    }
    receipt_path = run_dir / RECEIPT_NAME
    already_uploaded = matching_receipt(receipt_path, receipt_base)

    evaluation_steps = [step for step, data in rows if any(key.startswith("eval/") for key in data)]
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "already_uploaded": already_uploaded,
                "run_id": run_id,
                "rows": len(rows),
                "first_step": rows[0][0],
                "last_step": rows[-1][0],
                "evaluation_steps": evaluation_steps,
                "audit_accepted": accepted,
                "failed_checks": failed,
                "test_summary_included": test_summary is not None,
                "metrics_sha256": receipt_base["metrics_sha256"],
                "audit_sha256": receipt_base["audit_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run or already_uploaded:
        return 0

    actual_id, url = upload(
        args,
        rows,
        config,
        accepted,
        failed,
        test_summary["accepted"] if test_summary else None,
        run_id,
    )
    if actual_id != run_id:
        raise RuntimeError(f"W&B returned unexpected run id {actual_id!r}, expected {run_id!r}")
    receipt = {
        **receipt_base,
        "url": url,
        "row_count": len(rows),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "audit_accepted": accepted,
        "failed_checks": failed,
        "test_summary_included": test_summary is not None,
    }
    write_receipt(receipt_path, receipt)
    print(f"uploaded {len(rows)} steps to {url or args.project}; receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
