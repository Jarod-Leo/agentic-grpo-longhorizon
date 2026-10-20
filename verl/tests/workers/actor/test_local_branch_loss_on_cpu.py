"""CPU checks for experiment-C local clipped-PG normalization."""

import torch
from omegaconf import OmegaConf

from verl.workers.actor.dp_actor import (
    compute_local_branch_pg_loss,
    local_branch_microbatch_scale,
)


def make_config():
    return OmegaConf.create(
        {
            "policy_loss": {"loss_mode": "vanilla"},
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
        }
    )


def local_loss(log_prob, old_log_prob, advantages, mask):
    return compute_local_branch_pg_loss(
        log_prob=log_prob,
        fixed_old_log_prob=old_log_prob,
        local_advantages=advantages,
        local_mask=mask,
        config=make_config(),
    )[0]


def test_local_loss_masks_prefix_suffix_and_uses_fixed_old_logprob():
    log_prob = torch.tensor([[0.01, 0.02, 0.03, 0.04]], requires_grad=True)
    fixed_old = torch.zeros_like(log_prob)
    mask = torch.tensor([[0, 1, 1, 0]])
    advantages = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    loss = local_loss(log_prob, fixed_old, advantages, mask)
    loss.backward()

    assert log_prob.grad[0, 0].item() == 0
    assert log_prob.grad[0, 3].item() == 0
    assert log_prob.grad[0, 1:3].abs().sum().item() > 0
    current_as_old = local_loss(
        log_prob.detach(),
        log_prob.detach(),
        advantages,
        mask,
    )
    assert not torch.isclose(loss.detach(), current_as_old)


def test_dynamic_microbatches_match_full_combined_gradient_and_one_step():
    initial = torch.tensor(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.05, 0.06, 0.07, 0.08],
            [0.09, 0.10, 0.11, 0.12],
        ]
    )
    old = torch.zeros_like(initial)
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    advantages = torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0] * 4, [-0.5, 0.0, 0.0, 0.0]])

    full = initial.clone().requires_grad_()
    full_objective = full.square().mean() + 0.25 * local_loss(full, old, advantages, mask)
    full_objective.backward()
    expected_grad = full.grad.clone()

    split = initial.clone().requires_grad_()
    optimizer = torch.optim.SGD([split], lr=0.1)
    step_calls = 0
    for indices in (slice(0, 2), slice(2, 3)):
        root_part = split[indices].square().mean()
        root_scale = split[indices].shape[0] / split.shape[0]
        local_part = local_loss(
            split[indices],
            old[indices],
            advantages[indices],
            mask[indices],
        )
        local_scale = local_branch_microbatch_scale(mask[indices], mini_batch_actions=2)
        (root_part * root_scale + 0.25 * local_part * local_scale).backward()
    optimizer.step()
    step_calls += 1

    assert step_calls == 1
    assert torch.allclose(split.grad, expected_grad, atol=1e-7)
    assert torch.allclose(split.detach(), initial - 0.1 * expected_grad, atol=1e-7)


def test_zero_local_batch_is_differentiable_zero():
    log_prob = torch.randn(2, 3, requires_grad=True)
    loss = local_loss(
        log_prob,
        torch.zeros_like(log_prob),
        torch.zeros_like(log_prob),
        torch.zeros_like(log_prob),
    )
    loss.backward()

    assert loss.item() == 0
    assert torch.equal(log_prob.grad, torch.zeros_like(log_prob))
