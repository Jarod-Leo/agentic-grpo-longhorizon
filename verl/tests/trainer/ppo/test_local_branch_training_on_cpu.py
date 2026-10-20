"""CPU checks for local-branch rollout field alignment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ray")

from verl.trainer.ppo.ray_trainer import tensorize_local_branch_fields


def test_tensorize_local_fields_keeps_only_root_assistant_span():
    data = SimpleNamespace(
        batch={
            "responses": torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]]),
        },
        non_tensor_batch={
            "local_branch_mask": np.array([[0, 1, 1, 0], [0, 0, 0]], dtype=object),
            "local_branch_advantages": np.array(
                [[0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
                dtype=object,
            ),
            "local_branch_candidate0_token_ids": np.array([[11, 12], []], dtype=object),
            "local_branch_record_id": np.array(["root", None], dtype=object),
        },
    )

    tensorize_local_branch_fields(data)

    assert data.batch["local_branch_mask"].tolist() == [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert data.batch["local_branch_advantages"].tolist() == [
        [0.0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_tensorize_rejects_local_mask_on_tool_token():
    data = SimpleNamespace(
        batch={
            "responses": torch.tensor([[10, 11, 12]]),
            "response_mask": torch.tensor([[1, 0, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        non_tensor_batch={
            "local_branch_mask": np.array([[0, 1, 0]], dtype=object),
            "local_branch_advantages": np.array([[0.0, 1.0, 0.0]], dtype=object),
            "local_branch_candidate0_token_ids": np.array([[11]], dtype=object),
        },
    )

    with pytest.raises(ValueError, match="non-assistant"):
        tensorize_local_branch_fields(data)
