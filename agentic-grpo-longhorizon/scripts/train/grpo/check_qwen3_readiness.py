"""Check continuous/resumed state and LoRA updates on the allocated compute node."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_json(path: Path):
    return json.loads(path.read_text())


def equal_state(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(equal_state(left[k], right[k]) for k in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal_state(a, b) for a, b in zip(left, right))
        )
    return left == right


def check_checkpoint(checkpoint: Path, step: int) -> dict:
    """Validate saved adapter and optimizer state without loading base weights."""
    adapter = load_file(
        str(checkpoint / "actor/lora_adapter/adapter_model.safetensors")
    )
    assert adapter and all(
        torch.isfinite(value).all().item() for value in adapter.values()
    )
    config = load_json(checkpoint / "actor/lora_adapter/adapter_config.json")
    assert config.get("init_lora_weights", True) is True
    nonzero = sum(
        torch.count_nonzero(value).item()
        for key, value in adapter.items()
        if "lora_B" in key
    )
    extra = torch.load(
        checkpoint / "actor/extra_state_world_size_1_rank_0.pt",
        weights_only=False,
        map_location="cpu",
    )
    assert extra.get("rng") and extra.get("lr_scheduler")
    assert extra["lr_scheduler"]["last_epoch"] == step, extra["lr_scheduler"]
    optim = torch.load(
        checkpoint / "actor/optim_world_size_1_rank_0.pt",
        weights_only=False,
        map_location="cpu",
    )
    states = list(optim["state"].values())
    assert states and all(float(state["step"]) == step for state in states)
    assert all(
        torch.isfinite(value).all().item()
        for state in states
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return {
        "adapter_finite": True,
        "lora_b_nonzero": nonzero,
        "optimizer_scheduler_step": step,
    }


def check_single_run(run: Path, expected_lora_sha256: str | None = None) -> dict:
    """Gate a new method's formal run on a real update and complete checkpoint."""
    summary = load_json(run / "summary.json")
    assert summary["accepted"], "Run completeness failed"
    meta = summary["metadata"]
    assert (
        meta["mode"] == "train" and meta["start_step"] == 0 and meta["total_steps"] == 1
    )
    initializations = summary["lora_initialization"]
    hashes = {item["sha256"] for item in initializations}
    assert len(hashes) == 1, "Missing or inconsistent LoRA initialization"
    assert all(
        item.get("lora_b_numel", 0) > 0 and item.get("lora_b_nonzero") == 0
        for item in initializations
    ), "Missing evidence of zero-initialized LoRA B"
    if expected_lora_sha256 is not None:
        assert hashes == {expected_lora_sha256}, (
            "LoRA initialization differs from baseline"
        )
    checkpoint = Path(meta["checkpoint_root"]) / "global_step_1"
    checks = check_checkpoint(checkpoint, 1)
    checks["initial_lora_b_zero"] = True
    assert summary["learning"]["signal_observed"], (
        "No observed training reward variation"
    )
    assert checks["lora_b_nonzero"] > 0, "No observed LoRA parameter update"
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    actor = [row["data"] for row in metrics if "actor/grad_norm" in row.get("data", {})]
    assert len(actor) == 1 and actor[0]["actor/grad_norm"] > 0, (
        "No nonzero actor gradient"
    )
    if meta.get("checkpoint_archive_root"):
        assert checkpoint.is_symlink(), "Checkpoint not archived from SSD"
        expected = Path(meta["checkpoint_archive_root"]) / "global_step_1"
        assert checkpoint.resolve() == expected.resolve() and expected.is_dir()
        checks["hdd_archive_verified"] = True
    report = {
        "operational_checks_passed": True,
        "learning_signal_verified": True,
        "checks": checks,
        "lora_initialization_sha256": next(iter(hashes)),
        "training_reward_mode": meta.get("reward_mode", "binary"),
        "timing_s": {
            key.removeprefix("timing_s/"): value
            for key, value in actor[0].items()
            if key
            in {
                "timing_s/gen",
                "timing_s/update_actor",
                "timing_s/step",
                "timing_s/save_checkpoint",
            }
        },
        "gpu_hours": summary["gpu_hours"],
    }
    return report


def check_selector_resume(source: Path, restored: Path, run: Path, meta: dict) -> dict:
    """Replay the resumed outcome groups through the shared task selector."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.training.skill_bank import load_skill_bank
    from src.training.skill_selector import TaskSkillSelector

    teacher = meta["distillation"]["teacher"]
    initial, final = (
        load_json(source / "skill_selector.json"),
        load_json(restored / "skill_selector.json"),
    )
    for state, step in ((initial, 1), (final, 2)):
        assert state["version"] == "skill-selector-checkpoint-v1"
        assert state["global_step"] == state["selector"]["last_step"] == step
    bank = load_skill_bank(
        teacher["skill_bank_path"], teacher["skill_bank_sha256"], teacher["split_path"]
    )
    selector = TaskSkillSelector(
        bank,
        bank_sha256=teacher["skill_bank_sha256"],
        split_sha256=meta["split_sha256"],
        exploration_coef=teacher["ucb_exploration_coef"],
        mode=teacher["skill_selection"],
    )
    selector.load_state_dict(initial["selector"])
    outcomes = defaultdict(list)
    for line in (run / "eval_trajectories.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["validate"] is False and row["step"] == 2:
            assert row["split"] == "train" and row["score"] in (0, 1)
            outcomes[row["task_id"]].append(row["score"])
    audit_path = Path(
        teacher.get("selection_log_path") or run / "skill_selection.jsonl"
    )
    audit = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    records = {row["task_id"]: row for row in audit}
    assert outcomes and len(records) == len(audit) and records.keys() == outcomes.keys()
    selector.select(outcomes, 2)
    expected_audit = selector.commit(dict(outcomes), 2)
    for expected in expected_audit:
        record = records[expected["task_id"]]
        assert (
            record["global_step"] == 2
            and record["slurm_job_id"] == meta["slurm_job_id"]
        )
        assert all(record.get(key) == value for key, value in expected.items()), (
            "Resume skill audit differs from restored selection/outcomes"
        )
    assert selector.state_dict() == final["selector"], (
        "Resume selector state is not step-1 state plus outcome-group commits"
    )
    return {
        "skill_selector_resume_verified": True,
        "skill_selector_reward_groups": len(records),
    }


def check_resumed_run(first_run: Path, resume_run: Path) -> dict:
    """Verify one saved update and one independent resumed update through step two."""
    first, resume = [load_json(run / "summary.json") for run in (first_run, resume_run)]
    assert first["accepted"] and resume["accepted"], "Run completeness failed"
    first_meta, meta = first["metadata"], resume["metadata"]
    assert first_meta["mode"] == meta["mode"] == "train"
    assert first_meta["start_step"] == 0 and first_meta["total_steps"] == 1
    assert meta["start_step"] == 1 and meta["total_steps"] == 2
    source = Path(first_meta["checkpoint_root"]) / "global_step_1"
    checkpoint = Path(meta["checkpoint_root"]) / "global_step_2"
    assert Path(meta["resume_from"]).resolve() == source.resolve()
    checks = check_checkpoint(checkpoint, 2)
    assert checks["lora_b_nonzero"] > 0, "Resumed LoRA is still zero"
    log = (resume_run / "run.log").read_text(errors="replace")
    for marker in (
        "Setting global step to 1",
        "Loaded model from",
        "Loaded optimizer from",
        "Loaded rng from",
        "Loaded lr_scheduler from",
    ):
        assert marker in log, f"Missing resume evidence: {marker}"
    checks["checkpoint_restore_verified"] = True
    if meta.get("checkpoint_archive_root"):
        expected = Path(meta["checkpoint_archive_root"]) / "global_step_2"
        assert (
            checkpoint.is_symlink()
            and checkpoint.resolve() == expected.resolve()
            and expected.is_dir()
        )
        checks["hdd_archive_verified"] = True

    def uses_selector(metadata: dict) -> bool:
        distillation = metadata.get("distillation", {})
        actor, teacher = (
            distillation.get("actor", {}),
            distillation.get("teacher") or {},
        )
        return bool(
            actor.get("enabled")
            and actor.get("coef", 0) > 0
            and teacher.get("use_skills")
            and teacher.get("state_manager")
        )

    assert uses_selector(first_meta) == uses_selector(meta), (
        "Selector activation changed during resume"
    )
    if uses_selector(meta):
        checks.update(check_selector_resume(source, checkpoint, resume_run, meta))
    return {
        "operational_checks_passed": True,
        "resume_verified": True,
        "learning_signal_verified": first["learning"]["signal_observed"]
        and checks["lora_b_nonzero"] > 0,
        "checks": checks,
        "resume_from": str(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single-run", action="store_true")
    mode.add_argument(
        "--resume-run", type=Path, help="Check a separate step-1 to step-2 resume smoke"
    )
    parser.add_argument("--expected-lora-sha256")
    args = parser.parse_args()
    if args.resume_run is not None:
        report = check_resumed_run(args.root, args.resume_run)
        (args.resume_run / "readiness.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("READINESS", json.dumps(report))
        return
    if args.single_run:
        report = check_single_run(args.root, args.expected_lora_sha256)
        (args.root / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
        print("READINESS", json.dumps(report))
        return
    continuous, resume = args.root / "continuous", args.root / "resume"
    summaries = [load_json(run / "summary.json") for run in (continuous, resume)]
    assert all(s["accepted"] for s in summaries), "Run completeness failed"
    hashes = [
        {item["sha256"] for item in summary["lora_initialization"]}
        for summary in summaries
    ]
    assert len(hashes[0]) == 1 and hashes[0] == hashes[1], (
        "LoRA initialization mismatch"
    )
    checks = {"same_initial_adapter": True}
    for label, run in (("continuous", continuous), ("resume", resume)):
        checkpoint = run / "checkpoints/global_step_3"
        details = check_checkpoint(checkpoint, 3)
        checks.update({f"{label}_{key}": value for key, value in details.items()})
    saved = continuous / "checkpoints/global_step_2"
    extra = torch.load(
        saved / "actor/extra_state_world_size_1_rank_0.pt",
        weights_only=False,
        map_location="cpu",
    )
    assert extra.get("rng") and extra["lr_scheduler"]["last_epoch"] == 2
    log = (resume / "run.log").read_text(errors="replace")
    for marker in (
        "Setting global step to 2",
        "Loaded model from",
        "Loaded optimizer from",
        "Loaded rng from",
        "Loaded lr_scheduler from",
    ):
        assert marker in log, f"Missing resume evidence: {marker}"
    assert load_json(resume / "run.json")["resume_from"] == str(saved)
    tasks = []
    for run in (continuous, resume):
        rows = [
            json.loads(line)
            for line in (run / "eval_trajectories.jsonl").read_text().splitlines()
        ]
        tasks.append(Counter(row["task_id"] for row in rows if row["step"] == 3))
    assert tasks[0] == tasks[1], "Resumed next batch differs"
    data = [
        torch.load(
            run / "checkpoints/global_step_3/data.pt",
            weights_only=False,
            map_location="cpu",
        )
        for run in (continuous, resume)
    ]
    assert equal_state(*data), "Dataloader state diverged"
    checks["same_step3_tasks_and_dataloader"] = True
    learning = summaries[0]["learning"]["signal_observed"]
    if learning:
        assert checks["continuous_lora_b_nonzero"] > 0, (
            "Mixed rewards but no observed parameter update"
        )
    report = {
        "operational_checks_passed": True,
        "learning_signal_verified": learning
        and checks["continuous_lora_b_nonzero"] > 0,
        "checks": checks,
        "note": "MiMo outputs may differ after resume; rollout equality is not required.",
    }
    (args.root / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print("READINESS", json.dumps(report))


if __name__ == "__main__":
    main()
