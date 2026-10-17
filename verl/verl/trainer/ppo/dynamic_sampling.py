# Copyright 2026 The veRL authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Small, framework-independent helpers for DAPO-style group filtering."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DynamicSamplingState:
    """Counters that must survive checkpoint/resume."""

    generated_trajectories: int = 0
    optimizer_updates: int = 0
    skipped_updates: int = 0
    global_step: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "DynamicSamplingState":
        return cls(**{key: int(value.get(key, 0)) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class GroupFilterResult:
    """Indices and counts for complete groups with non-zero reward variance."""

    kept_indices: list[int]
    generated_groups: int
    mixed_groups: int


def filter_mixed_groups(
    uids: np.ndarray,
    metric_values: np.ndarray,
    expected_group_size: int,
) -> GroupFilterResult:
    """Keep complete groups whose metric has non-zero within-group standard deviation.

    Group order and trajectory order are preserved. A malformed partial group is rejected
    because mixing it into GRPO would change the intended number of samples per prompt.
    """
    if len(uids) != len(metric_values):
        raise ValueError("uid and dynamic-sampling metric must have the same number of trajectories")
    if expected_group_size <= 1:
        raise ValueError("dynamic sampling requires rollout.n > 1")

    group_indices: dict[object, list[int]] = {}
    group_values: dict[object, list[float]] = {}
    for index, (uid, value) in enumerate(zip(uids, metric_values, strict=True)):
        group_indices.setdefault(uid, []).append(index)
        group_values.setdefault(uid, []).append(float(value))

    kept_indices = []
    mixed_groups = 0
    for uid, indices in group_indices.items():
        if len(indices) != expected_group_size:
            raise ValueError(
                f"dynamic sampling expected {expected_group_size} trajectories for uid {uid!r}, got {len(indices)}"
            )
        if np.std(group_values[uid]) > 0:
            kept_indices.extend(indices)
            mixed_groups += 1

    return GroupFilterResult(
        kept_indices=kept_indices,
        generated_groups=len(group_indices),
        mixed_groups=mixed_groups,
    )


def can_generate_batch(state: DynamicSamplingState, trajectory_budget: int, batch_size: int) -> bool:
    """Return whether a whole rollout batch still fits in the trajectory budget."""
    if trajectory_budget <= 0:
        return True
    return state.generated_trajectories + batch_size <= trajectory_budget


def sampling_metrics(
    state: DynamicSamplingState,
    *,
    generated_this_step: int,
    num_gen_batches: int,
    generated_groups: int,
    mixed_groups: int,
    retained_groups: int,
    dropped_excess_groups: int,
    budget_exhausted: bool,
) -> dict[str, int]:
    prefix = "dynamic_sampling/"
    return {
        prefix + "generated_trajectories": state.generated_trajectories,
        prefix + "generated_this_step": generated_this_step,
        prefix + "num_gen_batches": num_gen_batches,
        prefix + "generated_groups": generated_groups,
        prefix + "mixed_groups": mixed_groups,
        prefix + "retained_groups": retained_groups,
        prefix + "dropped_excess_groups": dropped_excess_groups,
        prefix + "optimizer_updates": state.optimizer_updates,
        prefix + "skipped_updates": state.skipped_updates,
        prefix + "budget_exhausted": int(budget_exhausted),
    }
