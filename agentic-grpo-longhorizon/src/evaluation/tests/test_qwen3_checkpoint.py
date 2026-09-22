"""A formal run requires finite saved state and evidence of an actual update."""

import json
import runpy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

checks = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "scripts/train/grpo/check_qwen3_readiness.py"
    )
)


def checkpoint(tmp_path, value=0.2, step=1):
    path = tmp_path / "checkpoints/global_step_1"
    adapter = path / "actor/lora_adapter"
    adapter.mkdir(parents=True)
    save_file(
        {"layer.lora_B.weight": torch.tensor([[value]])},
        str(adapter / "adapter_model.safetensors"),
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"init_lora_weights": True})
    )
    torch.save(
        {"rng": {"cpu": [1]}, "lr_scheduler": {"last_epoch": step}},
        path / "actor/extra_state_world_size_1_rank_0.pt",
    )
    torch.save(
        {
            "state": {
                0: {"step": torch.tensor(float(step)), "exp_avg": torch.tensor([0.1])}
            }
        },
        path / "actor/optim_world_size_1_rank_0.pt",
    )
    return path


def test_checkpoint_verifier_accepts_finite_matching_step(tmp_path):
    result = checks["check_checkpoint"](checkpoint(tmp_path), 1)
    assert result["lora_b_nonzero"] == 1 and result["optimizer_scheduler_step"] == 1


@pytest.mark.parametrize("value,step", [(float("nan"), 1), (0.2, 2)])
def test_checkpoint_verifier_rejects_nan_or_wrong_optimizer_step(tmp_path, value, step):
    with pytest.raises(AssertionError):
        checks["check_checkpoint"](checkpoint(tmp_path, value, step), 1)


def test_single_update_gate_rejects_unchanged_adapter(tmp_path):
    path = checkpoint(tmp_path, value=0.0)
    summary = {
        "accepted": True,
        "metadata": {
            "mode": "train",
            "start_step": 0,
            "total_steps": 1,
            "checkpoint_root": str(path.parent),
        },
        "lora_initialization": [
            {"sha256": "initial", "lora_b_numel": 1, "lora_b_nonzero": 0}
        ],
        "learning": {"signal_observed": True},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(AssertionError, match="No observed LoRA parameter update"):
        checks["check_single_run"](tmp_path, "initial")


def test_single_update_accepts_prm_signal_in_saturated_outcome_group(tmp_path):
    path = checkpoint(tmp_path)
    summary = {
        "accepted": True,
        "metadata": {
            "mode": "train",
            "start_step": 0,
            "total_steps": 1,
            "checkpoint_root": str(path.parent),
            "reward_mode": "prm_lite",
        },
        "lora_initialization": [
            {"sha256": "initial", "lora_b_numel": 1, "lora_b_nonzero": 0}
        ],
        "learning": {
            "signal_observed": True,
            "mixed_outcome_groups": 0,
            "mixed_training_reward_groups": 1,
        },
        "gpu_hours": 0.1,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "data": {"actor/grad_norm": 0.2}}) + "\n"
    )
    result = checks["check_single_run"](tmp_path, "initial")
    assert (
        result["learning_signal_verified"] and result["checks"]["initial_lora_b_zero"]
    )
    summary["lora_initialization"][0]["lora_b_nonzero"] = 1
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(AssertionError, match="zero-initialized LoRA B"):
        checks["check_single_run"](tmp_path, "initial")
