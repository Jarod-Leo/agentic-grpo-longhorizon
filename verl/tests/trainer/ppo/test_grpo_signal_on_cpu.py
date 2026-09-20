# Copyright 2026 The veRL authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Check GRPO signal on binary groups with non-assistant token masking."""

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_binary_groups_have_zero_or_trainable_signal():
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 8)
    rewards = torch.zeros_like(mask)
    ids = np.array(["task"] * 8)
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, ids)
    assert torch.count_nonzero(advantages) == 0
    rewards[:3, 2] = 1.0
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, ids)
    assert torch.all(advantages[:3, 0] > 0)
    assert torch.all(advantages[3:, 0] < 0)
    assert torch.count_nonzero(advantages[:, [1, 3]]) == 0
    log_probs = torch.zeros_like(mask, requires_grad=True)
    loss = -(log_probs * advantages * mask).sum() / mask.sum()
    loss.backward()
    assert torch.isfinite(log_probs.grad).all()
    assert torch.count_nonzero(log_probs.grad) > 0
    assert torch.count_nonzero(log_probs.grad[:, [1, 3]]) == 0
