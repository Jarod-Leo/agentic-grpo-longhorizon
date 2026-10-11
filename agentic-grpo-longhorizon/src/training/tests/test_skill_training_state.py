"""CPU integration of pre-rollout choice, outcome groups and real checkpoint hooks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from src.training import distillation
from src.training.skill_bank import SKILL_VERSION, file_sha256
from src.training.tests.test_distillation import (
    Tokenizer,
    reply,
    skill_batch,
    skill_config,
)


class CandidateTokenizer(Tokenizer):
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if "candidate-a" in text:
            return [6, 5]
        assert "candidate-b" in text
        return [5, 6]


def selector_config(tmp_path: Path):
    config = skill_config(tmp_path)
    bank_path = Path(config.skill_bank_path)
    bank = json.loads(bank_path.read_text())
    template = bank["skills"][0]
    bank["version"] = SKILL_VERSION
    bank["skills"] = [
        {
            **template,
            "task_id": task_id,
            "skill_id": f"task-{task_id}-{candidate}",
            "source_trajectory_ids": [f"source-{task_id}-{candidate}"],
            "success_analysis": f"candidate-{candidate}",
        }
        for task_id in [0, 1]
        for candidate in ["a", "b"]
    ]
    bank_path.write_text(json.dumps(bank))
    config.update(
        {
            "skill_bank_sha256": file_sha256(bank_path),
            "state_manager": "src.training.distillation.SkillTeacherState",
            "skill_selection": "ucb",
            "ucb_exploration_coef": 2**0.5,
            "selection_log_path": str(tmp_path / "selection.jsonl"),
        }
    )
    return config


def repeated_outcome_batch():
    batch = skill_batch().repeat(8, interleave=True)
    batch.non_tensor_batch["outcome_reward"] = np.array(
        [1] * 4 + [0] * 4 + [1] * 8, dtype=float
    )
    # A PRM-shaped token reward must never replace the original binary outcome.
    batch.batch["token_level_scores"] = torch.full_like(
        batch.batch["old_log_probs"], 10.0
    )
    batch.reorder(torch.arange(len(batch) - 1, -1, -1))
    return batch


def trainer_stub(tmp_path: Path, config, *, resume: str | None = None):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"distillation": {"enabled": True, "coef": 0.001}},
                "rollout": {"temperature": 1.0},
            },
            "distillation": config,
            "trainer": {
                "default_local_dir": str(tmp_path / "ssd"),
                "default_hdfs_dir": None,
                "checkpoint_archive_dir": str(tmp_path / "archive"),
                "resume_mode": "resume_path" if resume else "disable",
                "resume_from_path": resume,
                "del_local_ckpt_after_load": False,
            },
        }
    )
    trainer.global_steps = 1
    trainer.use_critic = False
    trainer.tokenizer = CandidateTokenizer()
    trainer.loaded_actor_paths = []

    def save_actor(path, remote_path, step, max_ckpt_to_keep=None):
        root = Path(path)
        root.mkdir(parents=True)
        torch.save({"model_step": step}, root / "model.pt")

    def load_actor(path, **kwargs):
        trainer.loaded_actor_paths.append(path)
        if path is not None:
            assert (
                trainer._get_distillation_state().selector.last_step
                == trainer.global_steps
            )
            assert (
                torch.load(Path(path) / "model.pt", weights_only=False)["model_step"]
                == trainer.global_steps
            )

    trainer.actor_rollout_wg = SimpleNamespace(
        save_checkpoint=save_actor, load_checkpoint=load_actor
    )
    trainer.train_dataloader = SimpleNamespace(
        restored=None,
        state_dict=lambda: {"cursor": 17},
        load_state_dict=lambda state: setattr(
            trainer.train_dataloader, "restored", state
        ),
    )
    return trainer


def committed_trainer(tmp_path: Path):
    trainer = trainer_stub(tmp_path, selector_config(tmp_path))
    trainer._prepare_distillation(skill_batch())
    trainer._commit_distillation(repeated_outcome_batch())
    return trainer


def test_prepared_selection_stays_task_local_after_repeat_and_reorder(
    monkeypatch, tmp_path
):
    trainer = trainer_stub(tmp_path, selector_config(tmp_path))
    trainer._prepare_distillation(skill_batch())
    manager = trainer._get_distillation_state()
    assert manager.selected_for(0, 1)["skill_id"] == "task-0-a"
    batch = repeated_outcome_batch()
    original = {key: value.clone() for key, value in batch.batch.items()}
    requests = []
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda url, payload, timeout: requests.append(payload) or reply(payload),
    )
    trainer._score_distillation(batch)
    assert len(requests) == 16
    assert all(request["input_ids"][:2] == [6, 5] for request in requests)
    assert all(request["input_ids"][-3:] == [4, 5, 6] for request in requests[:8])
    assert all(request["input_ids"][-3:] == [3, 4, 5] for request in requests[8:])
    assert all(request["response_mask"] == [1, 0, 1] for request in requests)
    for key, value in original.items():
        torch.testing.assert_close(batch.batch[key], value)
    metrics = trainer._commit_distillation(batch)
    records = [json.loads(line) for line in manager.audit_path.read_text().splitlines()]
    assert len(records) == 2
    by_task = {record["task_id"]: record for record in records}
    assert by_task[0]["reward_mean"] == 0.5
    assert by_task[1]["reward_mean"] == 1.0
    assert all(
        record["reward_count"] == 8
        and record["count_before"] == 0
        and record["count_after"] == 1
        for record in records
    )
    assert all(
        record["global_step"] == 1 and "slurm_job_id" in record for record in records
    )
    assert metrics["distillation/skill_reward_mean"] == 0.75
    trainer.global_steps = 2
    trainer._prepare_distillation(skill_batch())
    assert manager.selected_for(0, 2)["skill_id"] == "task-0-b"
    assert manager.selected_for(1, 2)["skill_id"] == "task-1-b"


def test_missing_pre_rollout_selection_and_ucb_without_manager_are_rejected(
    monkeypatch, tmp_path
):
    config = selector_config(tmp_path)
    batch = skill_batch()
    batch.meta_info["distillation_policy_version"] = 0
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda *args: pytest.fail("Unprepared UCB must not contact teacher"),
    )
    with pytest.raises(ValueError, match="pre-rollout"):
        distillation.score_skill_teacher(batch, config, CandidateTokenizer())
    manager = distillation.SkillTeacherState(config)
    with pytest.raises(ValueError, match="not prepared"):
        distillation.score_skill_teacher(
            batch, config, CandidateTokenizer(), state_manager=manager
        )


@pytest.mark.parametrize("invalid", [None, 0.5, float("nan"), "1"])
def test_invalid_original_outcomes_do_not_commit_or_fall_back_to_token_scores(
    tmp_path, invalid
):
    trainer = trainer_stub(tmp_path, selector_config(tmp_path))
    trainer._prepare_distillation(skill_batch())
    batch = repeated_outcome_batch()
    if invalid is None:
        del batch.non_tensor_batch["outcome_reward"]
    else:
        batch.non_tensor_batch["outcome_reward"] = batch.non_tensor_batch[
            "outcome_reward"
        ].astype(object)
        batch.non_tensor_batch["outcome_reward"][0] = invalid
    with pytest.raises(ValueError, match="outcome_reward"):
        trainer._commit_distillation(batch)
    manager = trainer._get_distillation_state()
    assert manager.selector.last_step is None
    assert not manager.audit_path.exists()


def test_real_checkpoint_hooks_archive_selector_and_restore_before_actor(
    tmp_path, monkeypatch
):
    trainer = committed_trainer(tmp_path)
    manager = trainer._get_distillation_state()
    save = manager.save_checkpoint

    def checked_save(directory, step):
        assert (Path(directory) / "data.pt").is_file()
        assert not (tmp_path / "ssd/latest_checkpointed_iteration.txt").exists()
        save(directory, step)

    monkeypatch.setattr(manager, "save_checkpoint", checked_save)
    trainer._save_checkpoint()
    checkpoint = tmp_path / "ssd/global_step_1"
    assert checkpoint.is_symlink()
    assert (tmp_path / "archive/global_step_1/skill_selector.json").is_file()
    assert (tmp_path / "ssd/latest_checkpointed_iteration.txt").read_text() == "1"
    resumed = trainer_stub(
        tmp_path, trainer.config.distillation, resume=str(checkpoint)
    )
    resumed._load_checkpoint()
    assert resumed.loaded_actor_paths == [str(checkpoint / "actor")]
    assert resumed.train_dataloader.restored == {"cursor": 17}
    assert (
        resumed._get_distillation_state().selector.state_dict()
        == manager.selector.state_dict()
    )
    resumed.global_steps = 2
    resumed._prepare_distillation(skill_batch())
    assert (
        resumed._get_distillation_state().selected_for(0, 2)["skill_id"] == "task-0-b"
    )


def test_uncommitted_selection_cannot_publish_latest_checkpoint(tmp_path):
    trainer = trainer_stub(tmp_path, selector_config(tmp_path))
    trainer._prepare_distillation(skill_batch())
    with pytest.raises(ValueError, match="pending"):
        trainer._save_checkpoint()
    assert not (tmp_path / "ssd/latest_checkpointed_iteration.txt").exists()
    assert not (tmp_path / "archive/global_step_1").exists()
    assert not trainer._get_distillation_state().audit_path.exists()


@pytest.mark.parametrize(
    "case", ["missing", "step", "bank", "split", "mode", "coef", "data"]
)
def test_bad_selector_checkpoint_rejected_before_actor_load(tmp_path, case):
    trainer = committed_trainer(tmp_path)
    trainer._save_checkpoint()
    checkpoint = tmp_path / "ssd/global_step_1"
    path = checkpoint / "skill_selector.json"
    state = json.loads(path.read_text())
    if case == "missing":
        path.unlink()
    elif case == "data":
        (checkpoint / "data.pt").unlink()
    else:
        if case == "step":
            state["global_step"] = 2
        else:
            key = {
                "bank": "bank_sha256",
                "split": "split_sha256",
                "mode": "mode",
                "coef": "exploration_coef",
            }[case]
            state["selector"][key] = 9.0 if case == "coef" else "wrong"
        path.write_text(json.dumps(state))
    resumed = trainer_stub(
        tmp_path, trainer.config.distillation, resume=str(checkpoint)
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        resumed._load_checkpoint()
    assert resumed.loaded_actor_paths == []


@pytest.mark.parametrize("case", ["coef_zero", "skills_disabled", "no_factory"])
def test_disabled_state_has_no_bank_or_checkpoint_file_requirement(
    tmp_path, monkeypatch, case
):
    config = selector_config(tmp_path)
    trainer = trainer_stub(tmp_path, config)
    if case == "coef_zero":
        trainer.config.actor_rollout_ref.actor.distillation.coef = 0.0
    elif case == "skills_disabled":
        trainer.config.distillation.use_skills = False
    else:
        del trainer.config.distillation["state_manager"]
    monkeypatch.setattr(
        distillation,
        "SkillTeacherState",
        lambda *args: pytest.fail("Disabled state must not initialize"),
    )
    Path(config.skill_bank_path).unlink()
    assert trainer._get_distillation_state() is None
    trainer._prepare_distillation(skill_batch())
    assert trainer._commit_distillation(repeated_outcome_batch()) == {}
    trainer._save_checkpoint()
    assert not (tmp_path / "archive/global_step_1/skill_selector.json").exists()
