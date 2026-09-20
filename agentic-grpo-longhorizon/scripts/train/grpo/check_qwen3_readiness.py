"""Check continuous/resumed state and LoRA updates on the allocated compute node."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
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
        adapter = load_file(
            str(checkpoint / "actor/lora_adapter/adapter_model.safetensors")
        )
        assert adapter and all(
            torch.isfinite(value).all().item() for value in adapter.values()
        )
        config = load_json(checkpoint / "actor/lora_adapter/adapter_config.json")
        assert config.get("init_lora_weights", True) is True
        nonzero = sum(
            torch.count_nonzero(v).item() for k, v in adapter.items() if "lora_B" in k
        )
        checks[f"{label}_adapter_finite"] = True
        checks[f"{label}_lora_b_nonzero"] = nonzero
        extra = torch.load(
            checkpoint / "actor/extra_state_world_size_1_rank_0.pt",
            weights_only=False,
            map_location="cpu",
        )
        assert extra.get("rng") and extra.get("lr_scheduler")
        assert extra["lr_scheduler"]["last_epoch"] == 3, extra["lr_scheduler"]
        optim = torch.load(
            checkpoint / "actor/optim_world_size_1_rank_0.pt",
            weights_only=False,
            map_location="cpu",
        )
        states = list(optim["state"].values())
        assert states and all(float(state["step"]) == 3 for state in states)
        assert all(
            torch.isfinite(value).all().item()
            for state in states
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        checks[f"{label}_optimizer_scheduler_step"] = 3
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
