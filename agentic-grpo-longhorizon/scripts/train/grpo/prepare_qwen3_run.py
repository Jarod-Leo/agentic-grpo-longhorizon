"""Prepare train/test metadata and inputs; call inside an allocated job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from build_grpo_parquet import PROTOCOL, build_rows
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Applied Hydra configuration override",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    config_name = args.config_name or (
        "qwen3_mimo" if args.mode == "train" else "eval_qwen3"
    )
    config_dir = root / (
        "configs/train/grpo" if args.mode == "train" else "configs/eval/qwen3"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        method_config = compose(config_name=config_name, overrides=args.override)
    reward_mode = str(
        OmegaConf.select(method_config, "tau_bench_reward_mode", default="binary")
    )
    adv_estimator = str(OmegaConf.select(method_config, "algorithm.adv_estimator"))
    lata_alpha = OmegaConf.select(method_config, "algorithm.turn_discount.alpha")
    lata_alpha = float(lata_alpha) if lata_alpha is not None else None
    filter_config = OmegaConf.select(method_config, "algorithm.filter_groups")
    filter_config = (
        OmegaConf.to_container(filter_config, resolve=True)
        if filter_config is not None
        else {"enable": False}
    )
    dynamic_sampling = args.mode == "train" and bool(filter_config.get("enable"))
    assert reward_mode in {"binary", "prm_lite"}
    split_path = root / "experiments/sft_collect_airline/split.json"
    split = json.loads(split_path.read_text())
    train, test = split["seen_task_ids"], split["unseen_task_ids"]
    assert len(train) == 40 and len(test) == 10
    assert set(train).isdisjoint(test) and sorted(train + test) == list(range(50))
    skill_metadata = None
    teacher_config = OmegaConf.select(method_config, "distillation")
    if teacher_config is not None and teacher_config.get(
        "teacher_verify_revision", False
    ):
        from src.models.token_teacher import validate_local_model_revision

        validate_local_model_revision(
            teacher_config.teacher_model_path, teacher_config.teacher_revision
        )
    if teacher_config is not None and teacher_config.get("teacher_adapter_path"):
        from src.models.token_teacher import adapter_sha256

        expected = teacher_config.get("teacher_adapter_sha256")
        actual = adapter_sha256(teacher_config.teacher_adapter_path)
        if not expected or actual != expected:
            raise ValueError("Teacher adapter SHA256 mismatch before run preparation")
    if (
        teacher_config is not None
        and teacher_config.get("scorer")
        == "src.training.distillation.score_skill_teacher"
    ):
        from src.training.skill_bank import file_sha256, load_skill_bank

        configured_split = Path(teacher_config.split_path).resolve()
        if file_sha256(configured_split) != file_sha256(split_path):
            raise ValueError("Skill teacher split differs from run train/test split")
        use_skills = bool(teacher_config.use_skills)
        bank = (
            load_skill_bank(
                teacher_config.skill_bank_path,
                teacher_config.skill_bank_sha256,
                str(split_path),
            )
            if use_skills
            else {}
        )
        skill_metadata = {
            "use_skills": use_skills,
            "path": teacher_config.skill_bank_path,
            "sha256": teacher_config.skill_bank_sha256,
            "covered_train_task_ids": sorted(bank),
            "fallback_train_task_ids": sorted(set(train) - set(bank)),
            "coverage": len(bank) / len(train),
            "candidate_count": sum(len(candidates) for candidates in bank.values()),
            "candidates_per_task": {
                str(t): len(candidates) for t, candidates in bank.items()
            },
            "selection": str(teacher_config.get("skill_selection", "first")),
            "ucb_exploration_coef": float(
                teacher_config.get("ucb_exploration_coef", 2**0.5)
            ),
            "reward_association": "current_train_task_group_mean_binary_outcome",
            "prefix_max_tokens": int(teacher_config.skill_prefix_max_tokens),
        }
    start = 0
    dynamic_state = {
        "generated_trajectories": 0,
        "optimizer_updates": 0,
        "skipped_updates": 0,
        "global_step": 0,
    }
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
        if dynamic_sampling:
            state_path = args.resume / "dynamic_sampling.json"
            assert state_path.is_file(), "Dynamic-sampling checkpoint state is missing"
            dynamic_state = json.loads(state_path.read_text())
            assert dynamic_state.get("global_step") == start
            assert dynamic_state.get("optimizer_updates", 0) + dynamic_state.get(
                "skipped_updates", 0
            ) == start
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

    evaluation_steps = [] if dynamic_sampling else scheduled_steps(args.eval_freq)
    checkpoint_steps = [] if dynamic_sampling else scheduled_steps(args.save_freq)
    if dynamic_sampling:
        trajectory_budget = int(filter_config.get("trajectory_budget", 0))
        generated = int(dynamic_state["generated_trajectories"])
        assert filter_config.get("metric") == "outcome_reward"
        assert int(filter_config.get("max_num_gen_batches", 0)) > 0
        assert trajectory_budget > generated >= 0
        expected_training_trajectories = trajectory_budget - generated
        dynamic_metadata = {
            **filter_config,
            "enabled": True,
            "initial_generated_trajectories": generated,
            "initial_optimizer_updates": int(dynamic_state["optimizer_updates"]),
            "initial_skipped_updates": int(dynamic_state["skipped_updates"]),
            "expected_generated_trajectories_this_run": expected_training_trajectories,
            "schedule_policy": "frequency_plus_actual_final_step",
        }
    else:
        expected_training_trajectories = (args.steps - start) * 4 * args.samples
        dynamic_metadata = {"enabled": False}
    expected_trajectories = (
        len(ids) * args.samples
        if args.mode == "eval"
        else expected_training_trajectories
        + len(evaluation_steps) * len(ids) * args.eval_samples
    )
    metadata = {
        "protocol": PROTOCOL,
        "terminal_semantics": "first_environment_done_v1",
        "interaction_config_path": str(
            method_config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        ),
        "eval_temperature": float(method_config.actor_rollout_ref.rollout.val_kwargs.temperature),
        "eval_top_p": float(method_config.actor_rollout_ref.rollout.val_kwargs.top_p),
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
        "expected_trajectories": expected_trajectories,
        "expected_training_trajectories": expected_training_trajectories
        if args.mode == "train"
        else 0,
        "checkpoint_root": str(args.checkpoint_root or args.run / "checkpoints"),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_frequency": args.save_freq,
        "checkpoint_archive_root": os.environ.get("CHECKPOINT_ARCHIVE_ROOT"),
        "evaluation_steps": evaluation_steps,
        "evaluation_frequency": args.eval_freq,
        "eval_samples_per_task": args.eval_samples,
        "resume_from": str(args.resume) if args.resume else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "base_model": os.environ["POLICY_MODEL"],
        "seed": 42,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dynamic_sampling": dynamic_metadata,
    }
    if skill_metadata is not None:
        metadata["distillation"]["skill_bank"] = skill_metadata
    (args.run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"INPUTS_READY protocol={PROTOCOL} mode={args.mode} config={config_name} "
        f"reward={reward_mode} trajectories={metadata['expected_trajectories']}"
    )


if __name__ == "__main__":
    main()
