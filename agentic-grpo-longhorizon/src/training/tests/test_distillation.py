"""Driver/provider integration keeps original sampled tokens and fixed versions."""

import hashlib
import json
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from src.training import distillation
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class Tokenizer:
    chat_template = "fixed-template"
    special_tokens_map: ClassVar[dict] = {"eos_token": "eos"}
    all_special_ids: ClassVar[list] = [0]

    def get_vocab(self):
        return {"eos": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        assert "success_analysis: task-zero" in text
        return [6, 5]


def make_batch():
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6, 0]]),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]
            ),
            "responses": torch.tensor([[3, 4, 5, 0], [4, 5, 6, 0]]),
            "response_mask": torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]]),
            "old_log_probs": torch.full((2, 4), -2.0),
        },
        meta_info={"distillation_policy_version": 3},
    )


def config():
    return OmegaConf.create(
        {
            "scorer": "src.training.distillation.score_external_teacher",
            "endpoint": "http://127.0.0.1:1",
            "timeout_s": 1,
            "teacher_model_id": "tiny",
            "teacher_revision": "r1",
        }
    )


def reply(request):
    return {
        "input_sha256": hashlib.sha256(
            json.dumps(request["input_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "response_start": request["response_start"],
        "response_mask": request["response_mask"],
        "tokenizer_fingerprint": request["tokenizer_fingerprint"],
        "model_id": "tiny",
        "revision": "r1",
        "policy_version": request["policy_version"],
        "teacher_log_probs": [-1.5 if x else 0.0 for x in request["response_mask"]],
        **{
            key: request[key]
            for key in ("adapter_path", "adapter_sha256", "revision_verified")
            if key in request
        },
    }


def test_padding_roundtrip_preserves_real_user_and_tool_history(monkeypatch):
    batch = make_batch()
    original = batch.batch["input_ids"].clone()
    requests = []

    def respond(url, payload, timeout):
        requests.append(payload)
        return reply(payload)

    monkeypatch.setattr(distillation, "post_json", respond)
    scores, metrics = distillation.score_external_teacher(batch, config(), Tokenizer())
    assert requests[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert requests[0]["response_start"] == 2
    assert requests[0]["response_mask"] == [1, 0, 1]
    torch.testing.assert_close(scores, torch.tensor([[-1.5, 0, -1.5, 0]] * 2))
    assert metrics["distillation/scored_tokens"] == 4
    assert torch.equal(batch.batch["input_ids"], original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "wrong"),
        ("policy_version", 2),
        ("tokenizer_fingerprint", "wrong"),
        ("revision", "wrong"),
        ("teacher_log_probs", [-1.0]),
    ],
)
def test_teacher_response_cannot_silently_change_alignment_or_model(
    monkeypatch, field, value
):
    def respond(url, payload, timeout):
        return {**reply(payload), field: value}

    monkeypatch.setattr(distillation, "post_json", respond)
    with pytest.raises(ValueError):
        distillation.score_external_teacher(make_batch(), config(), Tokenizer())


def test_bad_response_suffix_or_padding_rejected():
    batch = make_batch()
    batch.batch["responses"][0, 0] = 6
    with pytest.raises(ValueError, match="suffix"):
        distillation.compact_teacher_request(batch, 0, "fingerprint")
    batch = make_batch()
    batch.batch["response_mask"][0, -1] = 1
    with pytest.raises(ValueError, match="padding"):
        distillation.compact_teacher_request(batch, 0, "fingerprint")


def test_real_trainer_hook_freezes_scores_at_current_batch_version(monkeypatch):
    monkeypatch.setattr(
        distillation, "post_json", lambda url, payload, timeout: reply(payload)
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"distillation": {"enabled": True, "coef": 1.0}},
                "rollout": {"temperature": 1.0},
            },
            "distillation": config(),
        }
    )
    trainer.tokenizer = Tokenizer()
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.global_steps = 4
    batch = make_batch()
    metrics = trainer._score_distillation(batch)
    assert metrics["distillation/policy_version"] == 3
    assert batch.batch["teacher_log_probs"].shape == batch.batch["responses"].shape
    assert not batch.batch["teacher_log_probs"].requires_grad
    trainer.config.actor_rollout_ref.actor.distillation.coef = 0.0
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda *a: pytest.fail("zero coefficient must not contact teacher"),
    )
    assert trainer._score_distillation(make_batch()) == {}


def skill_config(tmp_path, *, use_skills=True, with_skill=True):
    from src.training.skill_bank import file_sha256

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"seen_task_ids": [0, 1], "unseen_task_ids": [2]}))
    skill = {
        "skill_id": "fixed-zero",
        "task_id": 0,
        "source_trajectory_ids": ["success-0"],
        "success_analysis": "task-zero",
        "mistake_analysis": "No mistakes observed.",
        "golden_workflow": "Read current observations before calling tools.",
    }
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "version": "skill-bank-v1",
                "provenance": {"split_sha256": file_sha256(split)},
                "train_task_ids": [0, 1],
                "test_task_ids": [2],
                "skills": [skill] if with_skill else [],
            }
        )
    )
    result = config()
    result.update(
        {
            "scorer": "src.training.distillation.score_skill_teacher",
            "use_skills": use_skills,
            "split_path": str(split),
            "skill_bank_path": str(bank),
            "skill_bank_sha256": file_sha256(bank),
            "skill_prefix_max_tokens": 512,
            "teacher_max_length": 24576,
            "teacher_adapter_path": str(tmp_path / "frozen-e01"),
            "teacher_adapter_sha256": "fixed-e01-hash",
            "teacher_verify_revision": True,
        }
    )
    return result


def skill_batch():
    batch = make_batch()
    batch.non_tensor_batch["extra_info"] = np.array(
        [
            {
                "task_id": task_id,
                "split": "train",
                "interaction_kwargs": {"task_id": task_id},
            }
            for task_id in [0, 1]
        ],
        dtype=object,
    )
    return batch


def test_skill_teacher_prefix_is_task_local_and_student_tokens_stay_unchanged(
    monkeypatch, tmp_path
):
    batch = skill_batch()
    original = {key: value.clone() for key, value in batch.batch.items()}
    requests = []

    def respond(url, payload, timeout):
        requests.append(payload)
        return reply(payload)

    monkeypatch.setattr(distillation, "post_json", respond)
    scores, metrics = distillation.score_skill_teacher(
        batch, skill_config(tmp_path), Tokenizer()
    )
    assert requests[0]["input_ids"] == [6, 5, 1, 2, 3, 4, 5]
    assert requests[0]["response_start"] == 4
    assert requests[1]["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert requests[1]["response_start"] == 3
    assert all(request["response_mask"] == [1, 0, 1] for request in requests)
    assert all(request["adapter_sha256"] == "fixed-e01-hash" for request in requests)
    torch.testing.assert_close(scores, torch.tensor([[-1.5, 0, -1.5, 0]] * 2))
    assert metrics["distillation/skill_trajectories"] == 1
    assert metrics["distillation/skill_fallback_trajectories"] == 1
    assert metrics["distillation/skill_coverage"] == 0.5
    assert metrics["distillation/skill_tokens"] == 2
    for key, before in original.items():
        torch.testing.assert_close(batch.batch[key], before)


@pytest.mark.parametrize("use_skills,with_skill", [(False, True), (True, False)])
def test_no_skill_and_ablation_use_plain_frozen_teacher(
    monkeypatch, tmp_path, use_skills, with_skill
):
    requests = []
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda url, payload, timeout: requests.append(payload) or reply(payload),
    )
    monkeypatch.setattr(
        Tokenizer,
        "encode",
        lambda *args, **kwargs: pytest.fail("Plain teacher must not encode a skill"),
    )
    settings = skill_config(tmp_path, use_skills=use_skills, with_skill=with_skill)
    _, metrics = distillation.score_skill_teacher(skill_batch(), settings, Tokenizer())
    assert requests[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert requests[1]["input_ids"] == [1, 2, 3, 4, 5, 6]
    assert metrics["distillation/skill_coverage"] == 0.0
    assert metrics["distillation/skill_fallback_trajectories"] == 2


@pytest.mark.parametrize("case", ["test", "validate", "heldout_id", "nested_id"])
@pytest.mark.parametrize("use_skills", [True, False])
def test_skill_scorer_rejects_test_and_validation_before_http(
    monkeypatch, tmp_path, case, use_skills
):
    batch = skill_batch()
    if case == "test":
        batch.non_tensor_batch["extra_info"][0]["split"] = "test"
    elif case == "validate":
        batch.meta_info["validate"] = True
    elif case == "heldout_id":
        batch.non_tensor_batch["extra_info"][0]["task_id"] = 2
    else:
        batch.non_tensor_batch["extra_info"][0]["interaction_kwargs"]["task_id"] = 1
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda *args: pytest.fail("Invalid split must not contact teacher"),
    )
    with pytest.raises(ValueError):
        distillation.score_skill_teacher(
            batch, skill_config(tmp_path, use_skills=use_skills), Tokenizer()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter_sha256", "wrong"),
        ("adapter_path", "/other"),
        ("revision_verified", False),
    ],
)
def test_fixed_adapter_response_mismatch_is_rejected(
    monkeypatch, tmp_path, field, value
):
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda url, payload, timeout: {**reply(payload), field: value},
    )
    with pytest.raises(ValueError, match="identity"):
        distillation.score_skill_teacher(
            skill_batch(), skill_config(tmp_path), Tokenizer()
        )


def test_bank_hash_and_prefix_limits_fail_without_truncating(monkeypatch, tmp_path):
    settings = skill_config(tmp_path)
    monkeypatch.setattr(
        distillation,
        "post_json",
        lambda *args: pytest.fail("Invalid input must not contact teacher"),
    )
    settings.skill_bank_sha256 = "wrong"
    with pytest.raises(ValueError, match="SHA256"):
        distillation.score_skill_teacher(skill_batch(), settings, Tokenizer())
    from src.training.skill_bank import file_sha256

    settings.skill_bank_sha256 = file_sha256(settings.skill_bank_path)
    settings.teacher_max_length = 6
    with pytest.raises(ValueError, match="max_length"):
        distillation.score_skill_teacher(skill_batch(), settings, Tokenizer())
    settings.teacher_max_length = 24576
    monkeypatch.setattr(Tokenizer, "encode", lambda *args, **kwargs: [1] * 513)
    with pytest.raises(ValueError, match="513 tokens"):
        distillation.score_skill_teacher(skill_batch(), settings, Tokenizer())
