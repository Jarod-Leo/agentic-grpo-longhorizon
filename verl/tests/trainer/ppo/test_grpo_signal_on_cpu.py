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


async def generate_rollout(validate=False):
    """Use the real async worker orchestration with a tiny, local generation stub."""
    from omegaconf import OmegaConf

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase
    from verl.protocol import DataProto

    worker = object.__new__(AgentLoopWorkerBase)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 0.8,
                    "top_p": 1.0,
                    "calculate_log_probs": True,
                    "val_kwargs": {"temperature": 0.6, "top_p": 0.9},
                    "agent": {"default_agent_loop": "test"},
                }
            },
        }
    )
    sampled_temperatures = []

    async def run_agent(sampling_params, trajectory, **kwargs):
        sampled_temperatures.append(sampling_params["temperature"])
        return None

    worker._run_agent_loop = run_agent
    worker._postprocess = lambda outputs: DataProto.from_dict(
        tensors={"rollout_log_probs": torch.full((len(outputs), 2), -np.log(2))},
        meta_info={"reward_extra_keys": []},
    )
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.ones((2, 1), dtype=torch.long)},
        meta_info={"validate": validate},
    )
    output = await worker.generate_sequences(batch)
    return output, sampled_temperatures


def test_async_rollout_retains_validation_temperature():
    import asyncio

    output, sampled_temperatures = asyncio.run(generate_rollout(validate=True))
    assert sampled_temperatures == [0.6, 0.6]
    assert output.meta_info["temperature"] == 0.6


def test_async_bypass_without_kl_updates_actor_on_cpu(monkeypatch):
    import asyncio

    from omegaconf import OmegaConf

    from verl.protocol import DataProto
    from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
    from verl.workers.actor import dp_actor

    output, sampled_temperatures = asyncio.run(generate_rollout())
    assert sampled_temperatures == [0.8, 0.8]
    output = DataProto.concat([output, output])
    responses = torch.tensor([[0, 0], [1, 1]] * 2)
    mask = torch.tensor([[1.0, 0.0]] * 4)
    rewards = torch.zeros_like(mask)
    rewards[::2, 0] = 1
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, np.array([0, 0, 1, 1]))
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": mask,
            "advantages": advantages,
            "input_ids": torch.ones((4, 3), dtype=torch.long),
            "attention_mask": torch.ones((4, 3), dtype=torch.long),
            "position_ids": torch.arange(3).expand(4, -1),
        }
    ).union(output)
    config = OmegaConf.create(
        {
            "use_kl_loss": False,
            "use_dynamic_bsz": False,
            "entropy_coeff": 0.0,
            "ppo_mini_batch_size": 4,
            "ppo_micro_batch_size_per_gpu": 2,
            "ppo_epochs": 1,
            "loss_agg_mode": "token-mean",
            "grad_clip": 1.0,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss": {"loss_mode": "vanilla"},
        }
    )
    apply_rollout_correction(batch, OmegaConf.create({"bypass_mode": True}), config.policy_loss)
    assert "ref_log_prob" not in batch.batch
    torch.testing.assert_close(batch.batch["old_log_probs"], batch.batch["rollout_log_probs"])
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.config = config
    actor.actor_module = torch.nn.Linear(1, 2, bias=False)
    torch.nn.init.zeros_(actor.actor_module.weight)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.01)
    actor.scaler = None
    update_temperatures = []

    def forward(inputs, temperature, calculate_entropy=False):
        update_temperatures.append(temperature)
        logits = actor.actor_module.weight.T.expand(len(inputs["responses"]), -1) / temperature
        return None, logits.log_softmax(-1).gather(-1, inputs["responses"])

    actor._forward_micro_batch = forward
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    metrics = actor.update_policy(batch)
    assert update_temperatures == [0.8, 0.8]
    assert all(np.isfinite(metrics["actor/grad_norm"]))
    assert metrics["actor/grad_norm"][0] > 0
    assert torch.count_nonzero(actor.actor_module.weight) > 0


def test_dynamic_microbatch_preserves_per_trajectory_loss_and_gradient():
    """Variable-length trajectories retain the old microbatch=1 weighting."""
    from verl.trainer.ppo.core_algos import agg_loss

    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    values = torch.arange(16, dtype=torch.float64).reshape(4, 4).requires_grad_()
    reference = sum(agg_loss(values[i : i + 1], mask[i : i + 1], "token-mean") / 4 for i in range(4))
    expected_grad = torch.autograd.grad(reference, values)[0]
    for groups in [[[0, 1, 2, 3]], [[0, 2], [1, 3]], [[2], [0, 1, 3]]]:
        actual = sum(agg_loss(values[group], mask[group], "seq-mean-token-mean") * len(group) / 4 for group in groups)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(torch.autograd.grad(actual, values)[0], expected_grad)
