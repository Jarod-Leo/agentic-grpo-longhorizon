# Copyright 2026 The veRL authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Standalone validation reuses the trainer's evaluation without allocating an actor."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@pytest.fixture
def trainer(monkeypatch):
    instance = object.__new__(RayPPOTrainer)
    instance.config = OmegaConf.create(
        {
            "trainer": {
                "val_standalone": True,
                "val_only": True,
                "val_before_train": True,
                "resume_mode": "disable",
                "project_name": "test",
                "experiment_name": "standalone",
                "logger": [],
            },
            "actor_rollout_ref": {
                "model": {"lora_rank": 0, "lora_adapter_path": None},
                "rollout": {"mode": "async"},
            },
            "reward_model": {"enable": False},
        }
    )
    instance.resource_pool_manager = Mock()
    instance.use_critic = instance.use_reference_policy = instance.use_rm = False
    instance.val_reward_fn = Mock()
    manager = Mock()
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop", SimpleNamespace(AgentLoopManager=manager))
    return instance, manager


def test_standalone_initializes_only_rollout_and_runs_validation(trainer, monkeypatch):
    instance, manager = trainer
    instance.init_workers()
    manager.assert_called_once_with(config=instance.config)
    instance.resource_pool_manager.create_resource_pool.assert_not_called()
    assert instance.async_rollout_mode
    assert instance.async_rollout_manager is manager.return_value
    assert not hasattr(instance, "actor_rollout_wg")

    # Exercise fit's real checkpoint-disable and validation-only flow.
    instance._validate = Mock(return_value={"val/reward": 0.5})
    logger = Mock()
    monkeypatch.setattr("verl.utils.tracking.Tracking", Mock(return_value=logger))
    instance.fit()
    instance._validate.assert_called_once_with()
    logger.log.assert_called_once_with(data={"val/reward": 0.5}, step=0)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("trainer.val_only", False, "val_only"),
        ("trainer.val_before_train", False, "val_before_train"),
        ("trainer.resume_mode", "auto", "resume_mode"),
        ("actor_rollout_ref.rollout.mode", "sync", "async"),
        ("actor_rollout_ref.model.lora_rank", 16, "LoRA"),
        ("actor_rollout_ref.model.lora_adapter_path", "/tmp/adapter", "LoRA"),
        ("reward_model.enable", True, "rule-based"),
    ],
)
def test_standalone_rejects_incompatible_configs(trainer, key, value, message):
    instance, manager = trainer
    OmegaConf.update(instance.config, key, value)
    with pytest.raises(ValueError, match=message):
        instance.init_workers()
    manager.assert_not_called()
    instance.resource_pool_manager.create_resource_pool.assert_not_called()


@pytest.mark.parametrize("role", ["use_critic", "use_reference_policy", "use_rm"])
def test_standalone_rejects_additional_model_workers(trainer, role):
    instance, manager = trainer
    setattr(instance, role, True)
    with pytest.raises(ValueError, match="rule-based"):
        instance.init_workers()
    manager.assert_not_called()


def test_standalone_requires_validation_reward(trainer):
    instance, manager = trainer
    instance.val_reward_fn = None
    with pytest.raises(ValueError, match="val_reward_fn"):
        instance.init_workers()
    manager.assert_not_called()


def test_default_initialization_still_allocates_training_pool(trainer):
    instance, manager = trainer
    del instance.config.trainer.val_standalone
    instance.resource_pool_manager.create_resource_pool.side_effect = RuntimeError("original pool path")
    with pytest.raises(RuntimeError, match="original pool path"):
        instance.init_workers()
    instance.resource_pool_manager.create_resource_pool.assert_called_once_with()
    manager.assert_not_called()
    instance.actor_rollout_wg = Mock()
    assert instance._load_checkpoint() == 0
    instance.actor_rollout_wg.load_checkpoint.assert_called_once_with(None)
