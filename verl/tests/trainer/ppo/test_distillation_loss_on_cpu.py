# Copyright 2026 The veRL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU tests for sampled-token policy-gradient online distillation."""

import math
from types import MethodType

import pytest
import torch
from torch import nn

from verl import DataProto
from verl.trainer.ppo.distillation_loss import compute_sampled_token_opd_loss
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import DistillationConfig, FSDPActorConfig, OptimizerConfig


def opd_loss(current, old, teacher, mask):
    return compute_sampled_token_opd_loss(
        log_prob=current,
        old_log_prob=old,
        teacher_log_prob=teacher,
        response_mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )


def test_opd_freezes_teacher_and_old_and_has_expected_signal_direction():
    current = torch.zeros((1, 2), requires_grad=True)
    old = torch.zeros((1, 2), requires_grad=True)
    teacher = torch.tensor([[0.5, -0.5]], requires_grad=True)

    loss, metrics = opd_loss(current, old, teacher, torch.ones_like(current))
    loss.backward()

    assert old.grad is None
    assert teacher.grad is None
    assert current.grad[0, 0] < 0
    assert current.grad[0, 1] > 0
    assert metrics["actor/opd_positive_token_fraction"] == pytest.approx(0.5)
    assert metrics["actor/opd_negative_token_fraction"] == pytest.approx(0.5)


def test_opd_same_distribution_has_zero_signal_and_gradient():
    current = torch.tensor([[-0.4, -0.2]], requires_grad=True)
    old = current.detach().clone()
    teacher = old.clone()

    loss, metrics = opd_loss(current, old, teacher, torch.ones_like(current))
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(current.grad) == 0
    assert metrics["actor/opd_signal_token_fraction"] == pytest.approx(0.0)


def test_opd_clipping_respects_positive_and_negative_signal_boundaries():
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
    current = ratios.log().unsqueeze(0).requires_grad_()
    old = torch.zeros_like(current)
    teacher = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

    loss, metrics = opd_loss(current, old, teacher, torch.ones_like(current))
    loss.backward()

    # Positive signal clips above 1.2 but not below 0.8. Negative signal
    # clips below 0.8 but keeps the high-ratio penalty.
    assert current.grad[0, 0].item() == pytest.approx(0.0)
    assert current.grad[0, 1] < 0
    assert current.grad[0, 2].item() == pytest.approx(0.0)
    assert current.grad[0, 3] > 0
    assert metrics["actor/opd_clipfrac"] == pytest.approx(0.5)


def test_opd_ignores_tool_user_and_padding_positions_even_with_nan():
    current = torch.tensor([[0.0, math.nan, 0.0, math.nan]], requires_grad=True)
    old = torch.tensor([[0.0, math.nan, 0.0, math.nan]], requires_grad=True)
    teacher = torch.tensor([[0.3, math.nan, -0.2, math.nan]], requires_grad=True)
    assistant_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)

    loss, metrics = opd_loss(current, old, teacher, assistant_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(current.grad[~assistant_mask]) == 0
    assert old.grad is None
    assert teacher.grad is None
    assert metrics["actor/opd_signal_token_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("old", "teacher", "mask", "error"),
    [
        (torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 2), "shape"),
        (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), "active assistant"),
        (torch.zeros(1, 2), torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2), "finite"),
    ],
)
def test_opd_rejects_misaligned_empty_or_nonfinite_active_inputs(old, teacher, mask, error):
    with pytest.raises(ValueError, match=error):
        opd_loss(torch.zeros(1, 2, requires_grad=True), old, teacher, mask)


class FakeSampledPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampled_log_prob = nn.Parameter(torch.zeros(2))


def make_fake_actor(distillation, ppo_epochs=1):
    config = FSDPActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=ppo_epochs,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        entropy_coeff=0,
        use_kl_loss=False,
        distillation=distillation,
        optim=OptimizerConfig(lr=0.1),
    )
    actor = object.__new__(DataParallelPPOActor)
    actor.config = config
    actor.actor_module = FakeSampledPolicy()
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor.scaler = None
    actor.ulysses_sequence_parallel_size = 1

    def forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        batch_size = micro_batch["responses"].shape[0]
        return None, self.actor_module.sampled_log_prob.unsqueeze(0).expand(batch_size, -1)

    def optimizer_step(self):
        grad_norm = torch.linalg.vector_norm(self.actor_module.sampled_log_prob.grad.detach())
        self.actor_optimizer.step()
        return grad_norm

    actor._forward_micro_batch = MethodType(forward_micro_batch, actor)
    actor._optimizer_step = MethodType(optimizer_step, actor)
    return actor


def actor_batch(advantages, old_log_probs, teacher_log_probs=None):
    batch_size, response_length = old_log_probs.shape
    total_length = response_length + 1
    tensors = {
        "responses": torch.zeros((batch_size, response_length), dtype=torch.long),
        "response_mask": torch.ones((batch_size, response_length)),
        "input_ids": torch.zeros((batch_size, total_length), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, total_length), dtype=torch.long),
        "position_ids": torch.arange(total_length).expand(batch_size, -1),
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if teacher_log_probs is not None:
        tensors["teacher_log_probs"] = teacher_log_probs
    return DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0})


def test_pure_opd_actor_updates_with_all_fail_rl_and_keeps_batch_old_fixed(monkeypatch):
    actor = make_fake_actor(DistillationConfig(enabled=True, rl_coef=0.0, coef=1.0), ppo_epochs=2)
    old = torch.tensor([[0.4, 0.2], [0.3, 0.1]])
    teacher = old + 0.4
    data = actor_batch(torch.zeros_like(old), old, teacher)
    captured_old = []
    implementation = compute_sampled_token_opd_loss

    def capture_old(**kwargs):
        captured_old.append(kwargs["old_log_prob"].detach().clone())
        return implementation(**kwargs)

    monkeypatch.setattr(dp_actor_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(dp_actor_module, "compute_sampled_token_opd_loss", capture_old)
    monkeypatch.setattr(
        dp_actor_module,
        "get_policy_loss_fn",
        lambda *_args, **_kwargs: pytest.fail("pure OPD must not construct an RL policy loss"),
    )

    before = actor.actor_module.sampled_log_prob.detach().clone()
    metrics = actor.update_policy(data)

    expected_old = [old[0:1], old[1:2], old[0:1], old[1:2]]
    assert len(captured_old) == len(expected_old)
    for actual, expected in zip(captured_old, expected_old, strict=True):
        torch.testing.assert_close(actual, expected)
    assert not torch.equal(actor.actor_module.sampled_log_prob.detach(), before)
    assert all(value == pytest.approx(0.0) for value in metrics["actor/rl_loss"])
    assert all(value == pytest.approx(1.0) for value in metrics["actor/opd_signal_token_fraction"])
    assert all(value == pytest.approx(0.0) for value in metrics["actor/rl_logprob_grad_norm"])
    assert all(value > 0 for value in metrics["actor/opd_logprob_grad_norm"])
    assert all(value > 0 for value in metrics["actor/grad_norm"])


def test_zero_opd_coefficient_matches_disabled_rl_without_teacher(monkeypatch):
    monkeypatch.setattr(dp_actor_module, "get_device_id", lambda: "cpu")
    disabled = make_fake_actor(DistillationConfig())
    zero_opd = make_fake_actor(DistillationConfig(enabled=True, rl_coef=1.0, coef=0.0))
    old = torch.tensor([[0.4, 0.2], [0.3, 0.1]])
    advantages = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    data = actor_batch(advantages, old)

    disabled_metrics = disabled.update_policy(data)
    zero_opd_metrics = zero_opd.update_policy(data)

    torch.testing.assert_close(
        zero_opd.actor_module.sampled_log_prob.detach(), disabled.actor_module.sampled_log_prob.detach()
    )
    assert zero_opd_metrics["actor/pg_loss"] == pytest.approx(disabled_metrics["actor/pg_loss"])
    assert all(value == pytest.approx(0.0) for value in zero_opd_metrics["actor/opd_loss"])


def test_pure_opd_actor_neutralizes_masked_nan_outside_loss_helper(monkeypatch):
    monkeypatch.setattr(dp_actor_module, "get_device_id", lambda: "cpu")
    actor = make_fake_actor(DistillationConfig(enabled=True, rl_coef=0.0, coef=1.0))
    actor.actor_module.sampled_log_prob.data[1] = float("nan")
    old = torch.tensor([[0.2, math.nan], [0.3, math.nan]])
    teacher = torch.tensor([[0.6, math.nan], [0.7, math.nan]])
    data = actor_batch(torch.zeros_like(old), old, teacher)
    data.batch["response_mask"][:, 1] = 0

    before = actor.actor_module.sampled_log_prob[0].detach().clone()
    metrics = actor.update_policy(data)

    assert actor.actor_module.sampled_log_prob[0].detach() != before
    for name in (
        "actor/opd_loss",
        "actor/opd_logprob_grad_norm",
        "actor/rl_logprob_grad_norm",
        "actor/grad_norm",
    ):
        assert all(math.isfinite(value) for value in metrics[name])


def test_distillation_coefficients_must_be_finite_and_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        DistillationConfig(coef=-1.0)
    with pytest.raises(ValueError, match="finite"):
        DistillationConfig(rl_coef=float("nan"))
