# Copyright 2026 The veRL authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU checks for the configured LATA advantage used by the joint experiment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_lata_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage


def lata_config(alpha=1.05):
    return SimpleNamespace(turn_discount=SimpleNamespace(alpha=alpha))


def reference_lata(rewards, mask, ids, alpha=1.05, epsilon=1e-6):
    scores = rewards.sum(dim=-1)
    normalized = torch.empty_like(scores)
    for group in np.unique(ids):
        selected = np.flatnonzero(ids == group)
        values = scores[selected]
        normalized[selected] = (values - values.mean()) / (values.std() + epsilon)

    positions = torch.arange(mask.shape[1], dtype=torch.float64)
    result = torch.zeros_like(rewards)
    for row in range(mask.shape[0]):
        active = mask[row].bool()
        raw = torch.pow(torch.tensor(alpha, dtype=torch.float64), -positions[active])
        weights = raw * active.sum() / raw.sum()
        result[row, active] = normalized[row] * weights.to(result.dtype) / active.sum().sqrt()
    return result


def test_lata_matches_position_weighted_mean_one_reference():
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0]])
    rewards = torch.zeros_like(mask)
    rewards[1, 3] = 1.0
    ids = np.array(["task", "task"])

    actual, returns = compute_grpo_lata_outcome_advantage(rewards, mask, ids, config=lata_config())
    expected = reference_lata(rewards, mask, ids)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(returns, actual)
    assert torch.count_nonzero(actual[mask == 0]) == 0


def test_lata_is_finite_for_full_budget_with_tool_gaps_and_late_active_tokens():
    response_length = 12288
    mask = torch.zeros((2, response_length))
    mask[0, :32] = 1
    mask[0, 1000:1024] = 1
    mask[0, -16:] = 1
    mask[1, -2:] = 1
    rewards = torch.zeros_like(mask)
    rewards[1, -1] = 1.0

    advantages, _ = compute_grpo_lata_outcome_advantage(rewards, mask, np.array([0, 0]), config=lata_config())

    assert torch.isfinite(advantages).all()
    assert torch.count_nonzero(advantages[mask == 0]) == 0
    assert torch.count_nonzero(advantages[mask == 1]) > 0


def test_lata_alpha_changes_absolute_position_weighting():
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    rewards = torch.zeros_like(mask)
    rewards[1, 2] = 1.0
    ids = np.array([0, 0])

    uniform, _ = compute_grpo_lata_outcome_advantage(rewards, mask, ids, config=lata_config(alpha=1.0))
    discounted, _ = compute_grpo_lata_outcome_advantage(rewards, mask, ids, config=lata_config(alpha=2.0))

    torch.testing.assert_close(uniform[:, 0], uniform[:, 2])
    assert torch.all(discounted[:, 0].abs() > discounted[:, 2].abs())


def test_constant_rewards_have_zero_lata_advantage():
    mask = torch.tensor([[1.0, 0.0, 1.0]] * 4)
    rewards = torch.zeros_like(mask)
    rewards[:, 2] = 0.4
    advantages, _ = compute_grpo_lata_outcome_advantage(rewards, mask, np.array([0] * 4), config=lata_config())
    assert torch.count_nonzero(advantages) == 0


def test_prm_variation_creates_finite_pg_signal_when_outcomes_are_all_zero():
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)
    shaped_rewards = torch.zeros_like(mask)
    shaped_rewards[:, 2] = torch.tensor([-0.15, -0.03, 0.06, 0.15])
    advantages, _ = compute_grpo_lata_outcome_advantage(shaped_rewards, mask, np.array([0] * 4), config=lata_config())

    log_probs = torch.zeros_like(mask, requires_grad=True)
    loss = -(log_probs * advantages * mask).sum() / mask.sum()
    loss.backward()

    assert torch.isfinite(log_probs.grad).all()
    assert torch.count_nonzero(log_probs.grad[mask == 1]) > 0
    assert torch.count_nonzero(log_probs.grad[mask == 0]) == 0


def test_lata_rejects_empty_trajectories_and_invalid_alpha():
    rewards = torch.zeros((2, 3))
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ids = np.array([0, 0])
    with pytest.raises(ValueError, match="at least one active"):
        compute_grpo_lata_outcome_advantage(rewards, mask, ids, config=lata_config())
    with pytest.raises(ValueError, match="finite and positive"):
        compute_grpo_lata_outcome_advantage(rewards, torch.ones_like(mask), ids, config=lata_config(alpha=0.0))


def test_registered_grpo_lata_compute_advantage_route():
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    rewards = torch.zeros_like(mask)
    rewards[1, 2] = 1.0
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": mask},
        non_tensors={"uid": np.array(["task", "task"])},
    )

    result = compute_advantage(data, adv_estimator="grpo_lata", config=lata_config())
    direct, _ = compute_grpo_lata_outcome_advantage(rewards, mask, np.array(["task", "task"]), config=lata_config())
    torch.testing.assert_close(result.batch["advantages"], direct)
