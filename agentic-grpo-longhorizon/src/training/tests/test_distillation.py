"""Driver/provider integration keeps original sampled tokens and fixed versions."""

from types import SimpleNamespace
import hashlib
import json

import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from src.training import distillation


class Tokenizer:
    chat_template = "fixed-template"
    special_tokens_map = {"eos_token": "eos"}
    all_special_ids = [0]

    def get_vocab(self):
        return {"eos": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}


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
