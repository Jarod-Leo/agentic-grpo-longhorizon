# Copyright 2026 The veRL authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU checks for budgeted DAPO-style group filtering."""

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.dynamic_sampling import (
    DynamicSamplingState,
    can_generate_batch,
    filter_mixed_groups,
    sampling_metrics,
)
from verl.workers.actor.dp_actor import dynamic_microbatch_loss_scale


def test_filter_keeps_complete_mixed_groups_in_input_order():
    uids = np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8, dtype=object)
    rewards = np.array([0, 1] * 4 + [0] * 8 + [1] * 8)
    result = filter_mixed_groups(uids, rewards, expected_group_size=8)
    assert result.generated_groups == 3
    assert result.mixed_groups == 1
    assert result.kept_indices == list(range(8))


def test_filter_can_return_no_groups_and_rejects_partial_groups():
    import pytest

    result = filter_mixed_groups(np.array(["a"] * 8), np.zeros(8), expected_group_size=8)
    assert result.mixed_groups == 0
    assert result.kept_indices == []
    with pytest.raises(ValueError, match="expected 8 trajectories"):
        filter_mixed_groups(np.array(["a"] * 7), np.zeros(7), expected_group_size=8)


def test_budget_counts_every_generated_trajectory_and_restores_state():
    state = DynamicSamplingState(generated_trajectories=6368, optimizer_updates=71, skipped_updates=2, global_step=73)
    assert can_generate_batch(state, trajectory_budget=6400, batch_size=32)
    state.generated_trajectories += 32
    assert not can_generate_batch(state, trajectory_budget=6400, batch_size=32)
    assert DynamicSamplingState.from_dict(state.to_dict()) == state


def test_metrics_distinguish_attempts_from_optimizer_updates():
    state = DynamicSamplingState(
        generated_trajectories=96,
        optimizer_updates=1,
        skipped_updates=1,
        global_step=2,
    )
    metrics = sampling_metrics(
        state,
        generated_this_step=64,
        num_gen_batches=2,
        generated_groups=8,
        mixed_groups=3,
        retained_groups=3,
        dropped_excess_groups=0,
        budget_exhausted=False,
    )
    assert metrics["dynamic_sampling/generated_trajectories"] == 96
    assert metrics["dynamic_sampling/optimizer_updates"] == 1
    assert metrics["dynamic_sampling/skipped_updates"] == 1
    assert metrics["dynamic_sampling/retained_groups"] == 3


def test_short_dynamic_batch_is_normalized_by_its_actual_size():
    scales = [dynamic_microbatch_loss_scale(size, 8) for size in (3, 5)]
    assert sum(scales) == 1.0
    assert dynamic_microbatch_loss_scale(8, 8) == 1.0


def _rollout_batch(prefix: str, rewards_by_group: list[list[int]]) -> DataProto:
    rewards = np.array([reward for group in rewards_by_group for reward in group])
    uids = np.array(
        [f"{prefix}-{group_index}" for group_index in range(len(rewards_by_group)) for _ in range(8)],
        dtype=object,
    )
    return DataProto.from_dict(
        tensors={"token_level_scores": torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)},
        non_tensors={"uid": uids, "outcome_reward": rewards},
    )


def test_trainer_collector_accumulates_groups_and_drops_only_excess():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "metric": "outcome_reward",
                    "max_num_gen_batches": 3,
                    "trajectory_budget": 6400,
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 8}},
            "data": {"train_batch_size": 4, "gen_batch_size": 4},
        }
    )
    trainer.dynamic_sampling_state = DynamicSamplingState()
    outputs = [
        _rollout_batch("first", [[0, 1] * 4, [0, 1] * 4, [0] * 8, [1] * 8]),
        _rollout_batch("second", [[0, 1] * 4, [0, 1] * 4, [0, 1] * 4, [0] * 8]),
    ]

    def generate(*_args, **_kwargs):
        return outputs.pop(0), {"outcome_reward": []}

    trainer._generate_and_score_training_batch = generate
    batch, _, metrics, exhausted, consumed_epoch = trainer._collect_dynamic_training_batch(
        object(),
        iter([(1, object())]),
        {},
    )
    assert len(batch) == 32
    assert len(set(batch.non_tensor_batch["uid"])) == 4
    assert metrics["dynamic_sampling/num_gen_batches"] == 2
    assert metrics["dynamic_sampling/generated_groups"] == 8
    assert metrics["dynamic_sampling/mixed_groups"] == 5
    assert metrics["dynamic_sampling/dropped_excess_groups"] == 1
    assert trainer.dynamic_sampling_state.generated_trajectories == 64
    assert not exhausted
    assert consumed_epoch == 1


def test_trainer_collector_returns_none_after_three_uniform_batches():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "metric": "outcome_reward",
                    "max_num_gen_batches": 3,
                    "trajectory_budget": 96,
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 8}},
            "data": {"train_batch_size": 4, "gen_batch_size": 4},
        }
    )
    trainer.dynamic_sampling_state = DynamicSamplingState()
    outputs = [_rollout_batch(str(i), [[0] * 8] * 4) for i in range(3)]
    trainer._generate_and_score_training_batch = lambda *_args, **_kwargs: (outputs.pop(0), {})
    batch, _, metrics, exhausted, _ = trainer._collect_dynamic_training_batch(
        object(),
        iter([(0, object()), (0, object())]),
        {},
    )
    assert batch is None
    assert metrics["dynamic_sampling/retained_groups"] == 0
    assert metrics["dynamic_sampling/generated_trajectories"] == 96
    assert exhausted


def test_trainer_collector_flushes_a_short_batch_at_generation_limit():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "filter_groups": {
                    "metric": "outcome_reward",
                    "max_num_gen_batches": 2,
                    "trajectory_budget": 6400,
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 8}},
            "data": {"train_batch_size": 4, "gen_batch_size": 4},
        }
    )
    trainer.dynamic_sampling_state = DynamicSamplingState(generated_trajectories=32, global_step=1)
    outputs = [
        _rollout_batch("epoch-0", [[0, 1] * 4, [0] * 8, [0] * 8, [1] * 8]),
        _rollout_batch("epoch-1", [[0, 1] * 4, [0] * 8, [1] * 8, [1] * 8]),
    ]
    trainer._generate_and_score_training_batch = lambda *_args, **_kwargs: (outputs.pop(0), {})
    batch, _, metrics, exhausted, consumed_epoch = trainer._collect_dynamic_training_batch(
        object(),
        iter([(1, object())]),
        {},
    )
    assert len(batch) == 16
    assert metrics["dynamic_sampling/retained_groups"] == 2
    assert metrics["dynamic_sampling/generated_trajectories"] == 96
    assert consumed_epoch == 1
    assert not exhausted


def _fit_harness(monkeypatch):
    from verl.trainer.ppo import ray_trainer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    logged = []
    saved = []
    validated = []

    class FakeTracking:
        def __init__(self, **_kwargs):
            pass

        def log(self, data, step):
            logged.append((step, dict(data)))

    class FakeProgress:
        def __init__(self, **_kwargs):
            pass

        def update(self, _value):
            pass

        def close(self):
            pass

    monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
    monkeypatch.setattr(ray_trainer, "tqdm", FakeProgress)
    monkeypatch.setattr(ray_trainer, "should_save_ckpt_esi", lambda **_kwargs: False)
    monkeypatch.setattr(ray_trainer, "compute_advantage", lambda batch, **_kwargs: batch)
    monkeypatch.setattr(ray_trainer, "compute_data_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(ray_trainer, "compute_timing_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(ray_trainer, "compute_throughout_metrics", lambda **_kwargs: {})

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "dynamic",
                "logger": ["console"],
                "val_before_train": False,
                "balance_batch": False,
                "total_epochs": 2,
                "test_freq": 100,
                "save_freq": 50,
                "critic_warmup": 0,
                "esi_redundant_time": 0,
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "rollout_correction": None,
                "filter_groups": {
                    "enable": True,
                    "metric": "outcome_reward",
                    "max_num_gen_batches": 3,
                    "trajectory_budget": 32,
                },
            },
            "actor_rollout_ref": {
                "actor": {"loss_agg_mode": "token-mean"},
                "rollout": {"n": 8, "skip_rollout": False, "multi_turn": {"enable": True}},
            },
        }
    )
    trainer.dynamic_sampling_state = DynamicSamplingState()
    trainer.total_training_steps = 200
    trainer.val_reward_fn = object()
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.train_dataset = object()
    trainer.train_dataloader = SimpleNamespace(sampler=None)
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer._load_checkpoint = lambda: None
    trainer._training_batches = lambda: iter([(0, object())])
    trainer._start_profiling = lambda *_args, **_kwargs: None
    trainer._stop_profiling = lambda *_args, **_kwargs: None
    trainer._score_distillation = lambda _batch: {}
    trainer._commit_distillation = lambda _batch: {}
    trainer._save_checkpoint = lambda: saved.append(trainer.global_steps)
    trainer._validate = lambda: validated.append(trainer.global_steps) or {"val/reward": 0.5}
    return trainer, logged, saved, validated


def test_fit_zero_group_budget_end_still_validates_and_saves(monkeypatch):
    trainer, logged, saved, validated = _fit_harness(monkeypatch)

    def collect(*_args, **_kwargs):
        trainer.dynamic_sampling_state.generated_trajectories = 32
        return (
            None,
            {},
            sampling_metrics(
                trainer.dynamic_sampling_state,
                generated_this_step=32,
                num_gen_batches=1,
                generated_groups=4,
                mixed_groups=0,
                retained_groups=0,
                dropped_excess_groups=0,
                budget_exhausted=True,
            ),
            True,
            None,
        )

    trainer._collect_dynamic_training_batch = collect
    trainer.fit()
    assert saved == [1]
    assert validated == [1]
    assert trainer.dynamic_sampling_state.optimizer_updates == 0
    assert trainer.dynamic_sampling_state.skipped_updates == 1
    assert logged[-1][1]["dynamic_sampling/budget_exhausted"] == 1
    assert logged[-1][1]["dynamic_sampling/skipped_updates"] == 1


def test_fit_short_batch_updates_actor_once_and_logs_actual_counts(monkeypatch):
    trainer, logged, saved, validated = _fit_harness(monkeypatch)
    rewards = torch.tensor([[1.0], [0.0]] * 4)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((8, 3), dtype=torch.long),
            "attention_mask": torch.ones((8, 3), dtype=torch.long),
            "responses": torch.ones((8, 1), dtype=torch.long),
            "response_mask": torch.ones((8, 1)),
            "token_level_scores": rewards,
        },
        non_tensors={"uid": np.array(["mixed"] * 8), "outcome_reward": rewards.squeeze(-1).numpy()},
    )

    class ActorWorker:
        def __init__(self):
            self.updated_sizes = []

        def compute_log_prob(self, actor_batch):
            return DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.zeros_like(actor_batch.batch["response_mask"]),
                    "entropys": torch.zeros_like(actor_batch.batch["response_mask"]),
                }
            )

        def update_actor(self, actor_batch):
            self.updated_sizes.append(len(actor_batch))
            return DataProto(meta_info={"metrics": {"actor/pg_loss": [0.25]}})

    trainer.actor_rollout_wg = ActorWorker()

    def collect(*_args, **_kwargs):
        trainer.dynamic_sampling_state.generated_trajectories = 32
        return (
            batch,
            {"outcome_reward": batch.non_tensor_batch["outcome_reward"].tolist()},
            sampling_metrics(
                trainer.dynamic_sampling_state,
                generated_this_step=32,
                num_gen_batches=1,
                generated_groups=4,
                mixed_groups=1,
                retained_groups=1,
                dropped_excess_groups=0,
                budget_exhausted=True,
            ),
            True,
            None,
        )

    trainer._collect_dynamic_training_batch = collect
    trainer.fit()
    assert trainer.actor_rollout_wg.updated_sizes == [8]
    assert trainer.dynamic_sampling_state.optimizer_updates == 1
    assert trainer.dynamic_sampling_state.skipped_updates == 0
    assert saved == [1]
    assert validated == [1]
    assert logged[-1][1]["dynamic_sampling/retained_groups"] == 1
    assert logged[-1][1]["dynamic_sampling/optimizer_updates"] == 1
