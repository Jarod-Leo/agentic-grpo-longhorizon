# Copyright 2026 The veRL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Losses for online policy distillation."""

import math
from typing import Any

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def _validate_opd_inputs(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    expected_shape = log_prob.shape
    for name, value in (
        ("old_log_prob", old_log_prob),
        ("teacher_log_prob", teacher_log_prob),
        ("response_mask", response_mask),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match log_prob shape {tuple(expected_shape)}")

    if response_mask.is_floating_point() and not torch.isfinite(response_mask).all():
        raise ValueError("response_mask must contain only finite values")
    active_mask = response_mask.bool()
    if not active_mask.any():
        raise ValueError("OPD requires at least one active assistant token")

    for name, value in (
        ("log_prob", log_prob),
        ("old_log_prob", old_log_prob),
        ("teacher_log_prob", teacher_log_prob),
    ):
        if not torch.isfinite(value[active_mask]).all():
            raise ValueError(f"{name} must be finite at active assistant tokens")
    return active_mask


def compute_sampled_token_opd_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped sampled-token policy-gradient OPD loss.

    The fixed distillation signal for each sampled response token is the
    stopped-gradient difference between teacher and old-policy log-probability.
    This objective only uses log-probabilities for the sampled token; it is not
    an exact full-vocabulary KL divergence.
    """
    for name, value in (("clip_ratio_low", clip_ratio_low), ("clip_ratio_high", clip_ratio_high)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if clip_ratio_low >= 1:
        raise ValueError(f"clip_ratio_low must be less than 1, got {clip_ratio_low}")

    active_mask = _validate_opd_inputs(log_prob, old_log_prob, teacher_log_prob, response_mask)
    loss_mask = active_mask.to(dtype=log_prob.dtype)

    # Neutralize masked NaNs before arithmetic so tool, user, and padding
    # positions cannot contaminate the sampled assistant-token objective.
    current = torch.where(active_mask, log_prob, torch.zeros_like(log_prob))
    old = torch.where(active_mask, old_log_prob.detach(), torch.zeros_like(old_log_prob))
    teacher = torch.where(active_mask, teacher_log_prob.detach(), torch.zeros_like(teacher_log_prob))
    distillation_advantage = (teacher - old).detach()

    log_ratio = torch.clamp(current - old, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    surrogate = ratio * distillation_advantage
    clipped_surrogate = clipped_ratio * distillation_advantage
    loss_mat = -torch.minimum(surrogate, clipped_surrogate)
    loss = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    signal_mask = distillation_advantage.ne(0)
    metrics = {
        "actor/opd_clipfrac": verl_F.masked_mean((clipped_surrogate < surrogate).to(log_prob.dtype), loss_mask)
        .detach()
        .item(),
        "actor/opd_signal_token_fraction": verl_F.masked_mean(signal_mask.to(log_prob.dtype), loss_mask)
        .detach()
        .item(),
        "actor/opd_positive_token_fraction": verl_F.masked_mean(
            (distillation_advantage > 0).to(log_prob.dtype), loss_mask
        )
        .detach()
        .item(),
        "actor/opd_negative_token_fraction": verl_F.masked_mean(
            (distillation_advantage < 0).to(log_prob.dtype), loss_mask
        )
        .detach()
        .item(),
        "actor/opd_advantage_mean": verl_F.masked_mean(distillation_advantage, loss_mask).detach().item(),
    }
    return loss, metrics


def logprob_branch_grad_norm(loss: torch.Tensor, log_prob: torch.Tensor) -> float:
    """Return a branch loss gradient norm with respect to sampled-token log-probability.

    This diagnostic stops at the log-probability tensor and therefore is not a
    model-parameter gradient norm and does not perform a second model backward.
    """
    if not loss.requires_grad or not log_prob.requires_grad:
        return 0.0
    (gradient,) = torch.autograd.grad(loss, log_prob, retain_graph=True, allow_unused=True)
    if gradient is None:
        return 0.0
    return torch.linalg.vector_norm(gradient.detach().float()).item()


__all__ = ["compute_sampled_token_opd_loss", "logprob_branch_grad_norm"]
