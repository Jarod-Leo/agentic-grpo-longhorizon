"""Prepare train/test metadata and inputs; call inside an allocated job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from build_grpo_parquet import PROTOCOL, build_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=-1)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--config-name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config_name = args.config_name or (
        "qwen3_mimo" if args.mode == "train" else "eval_qwen3"
    )
    config_dir = root / (
        "configs/train/grpo" if args.mode == "train" else "configs/eval/qwen3"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        method_config = compose(config_name=config_name)
    reward_mode = str(
        OmegaConf.select(method_config, "tau_bench_reward_mode", default="binary")
    )
    adv_estimator = str(OmegaConf.select(method_config, "algorithm.adv_estimator"))
    lata_alpha = OmegaConf.select(method_config, "algorithm.turn_discount.alpha")
    lata_alpha = float(lata_alpha) if lata_alpha is not None else None
    assert reward_mode in {"binary", "prm_lite"}
    split_path = root / "experiments/sft_collect_airline/split.json"
    split = json.loads(split_path.read_text())
    train, test = split["seen_task_ids"], split["unseen_task_ids"]
    assert len(train) == 40 and len(test) == 10
    assert set(train).isdisjoint(test) and sorted(train + test) == list(range(50))
    start = 0
    if args.resume:
        start = int(args.resume.name.removeprefix("global_step_"))
        for rel in (
            "data.pt",
            "actor/model_world_size_1_rank_0.pt",
            "actor/optim_world_size_1_rank_0.pt",
            "actor/extra_state_world_size_1_rank_0.pt",
        ):
            assert (args.resume / rel).is_file(), f"Incomplete checkpoint: {rel}"
        assert 0 < start < args.steps
    if args.adapter:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            assert (args.adapter / name).is_file(), f"Missing adapter file: {name}"
    pd.DataFrame(build_rows(train, "train")).to_parquet(
        args.run / "train.parquet", index=False
    )
    ids = train if args.split == "train" else test
    pd.DataFrame(build_rows(ids, args.split)).to_parquet(
        args.run / "eval.parquet", index=False
    )

    def scheduled_steps(frequency: int) -> list[int]:
        if args.mode != "train" or frequency <= 0:
            return []
        return [
            step
            for step in range(start + 1, args.steps + 1)
            if step % frequency == 0 or step == args.steps
        ]

    evaluation_steps = scheduled_steps(args.eval_freq)
    metadata = {
        "protocol": PROTOCOL,
        "config_name": config_name,
        "reward_mode": reward_mode,
        "adv_estimator": adv_estimator,
        "lata_alpha": lata_alpha,
        "prm_coefficient": 0.3 if reward_mode == "prm_lite" else 0.0,
        "distillation": {
            "actor": OmegaConf.to_container(
                method_config.actor_rollout_ref.actor.distillation, resolve=True
            )
            if "distillation" in method_config.actor_rollout_ref.actor
            else {"enabled": False},
            "teacher": OmegaConf.to_container(method_config.distillation, resolve=True)
            if "distillation" in method_config
            else None,
        },
        "mode": args.mode,
        "eval_split": args.split,
        "train_task_ids": train,
        "test_task_ids": test,
        "samples_per_task": args.samples,
        "train_batch_size": 4,
        "total_steps": args.steps,
        "start_step": start,
        "expected_trajectories": len(ids) * args.samples
        if args.mode == "eval"
        else (args.steps - start) * 4 * args.samples
        + len(evaluation_steps) * len(ids) * args.eval_samples,
        "checkpoint_root": str(args.checkpoint_root or args.run / "checkpoints"),
        "checkpoint_steps": scheduled_steps(args.save_freq),
        "checkpoint_archive_root": os.environ.get("CHECKPOINT_ARCHIVE_ROOT"),
        "evaluation_steps": evaluation_steps,
        "eval_samples_per_task": args.eval_samples,
        "resume_from": str(args.resume) if args.resume else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "base_model": os.environ["POLICY_MODEL"],
        "seed": 42,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"INPUTS_READY protocol={PROTOCOL} mode={args.mode} config={config_name} "
        f"reward={reward_mode} trajectories={metadata['expected_trajectories']}"
    )


if __name__ == "__main__":
    main()
