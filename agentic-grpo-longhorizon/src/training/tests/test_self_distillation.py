"""CPU tests for feedback-conditioned pre-update self-teacher scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from verl.protocol import DataProto

from src.training.self_distillation import build_self_teacher_batch, score_self_teacher


class CharTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.encoded_texts = []

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        self.encoded_texts.append(text)
        return [10 + ord(char) % 40 for char in text]


class TinyCausalModel(nn.Module):
    """Small causal backend whose selected-token scores depend on every prefix token."""

    def __init__(self, vocab_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.adapter_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, input_ids, attention_mask):
        state = (input_ids.float() * attention_mask).cumsum(dim=-1)
        centers = torch.remainder(state * (1 + self.adapter_scale), self.vocab_size)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float32)
        return -(centers.unsqueeze(-1) - vocabulary).abs() / 5


class TinyActorWorkerGroup:
    def __init__(self):
        self.model = TinyCausalModel()
        self.compute_calls = 0
        self.meta_info_seen = []
        self.teacher_batches = []

    def compute_log_prob(self, data):
        self.compute_calls += 1
        self.meta_info_seen.append(dict(data.meta_info))
        self.teacher_batches.append(data)
        assert data.meta_info["temperature"] == 1.0
        assert "is_lora" not in data.meta_info

        responses = data.batch["responses"]
        response_width = responses.shape[1]
        response_start = data.batch["input_ids"].shape[1] - response_width
        with torch.no_grad():
            logits = self.model(data.batch["input_ids"], data.batch["attention_mask"])
            shifted = logits[:, response_start - 1 : -1]
            log_probs = (
                torch.log_softmax(shifted, dim=-1)
                .gather(dim=-1, index=responses.unsqueeze(-1))
                .squeeze(-1)
            )
        return DataProto.from_dict(
            tensors={
                "old_log_probs": log_probs,
                "entropys": torch.zeros_like(log_probs),
            }
        )


class CorruptActorWorkerGroup(TinyActorWorkerGroup):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def compute_log_prob(self, data):
        output = super().compute_log_prob(data)
        output.batch["old_log_probs"][0, 0] = self.value
        return output


def feedback(mode="F2", text="The inventory tool returned an error.", rules=None):
    if rules is None:
        rules = ["tool_error"] if mode == "F2" else []
    return {
        "version": "opsd-f2-v1",
        "mode": mode,
        "text": text if mode == "F2" else "",
        "rules": rules,
    }


def make_batch(feedbacks=None, policy_version=7):
    if feedbacks is None:
        feedbacks = [
            feedback(),
            feedback(text="The same failing tool call was repeated."),
        ]
    input_ids = torch.tensor([[0, 0, 1, 2, 3, 4, 0], [0, 5, 6, 2, 7, 8, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0]])
    position_ids = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": torch.tensor([[3, 4, 0], [7, 8, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "old_log_probs": torch.full((2, 3), -2.0),
        },
        non_tensors={"opsd_feedback": np.array(feedbacks, dtype=object)},
        meta_info={"distillation_policy_version": policy_version, "is_lora": True},
    )


def config(mode="F2", feedback_max_tokens=512, max_teacher_length=24576):
    return SimpleNamespace(
        teacher_mode="self_feedback",
        feedback_mode=mode,
        feedback_version="opsd-f2-v1",
        feedback_max_tokens=feedback_max_tokens,
        max_teacher_length=max_teacher_length,
    )


def test_f0_teacher_batch_is_an_exact_cloned_identity_without_tokenization():
    batch = make_batch([feedback("F0"), feedback("F0")])
    tokenizer = CharTokenizer()

    teacher_batch, metrics = build_self_teacher_batch(
        batch, config(mode="F0"), tokenizer
    )

    for name in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
    ):
        assert teacher_batch.batch[name].shape == batch.batch[name].shape
        assert torch.equal(teacher_batch.batch[name], batch.batch[name])
        assert teacher_batch.batch[name].data_ptr() != batch.batch[name].data_ptr()
    assert tokenizer.encoded_texts == []
    assert "is_lora" not in teacher_batch.meta_info
    assert metrics == {
        "distillation/feedback_trajectories": 0,
        "distillation/feedback_tokens": 0,
        "distillation/feedback_coverage": 0.0,
    }


def test_f2_prefix_shifts_only_teacher_prompt_and_keeps_response_alignment():
    batch = make_batch()
    original = {name: value.clone() for name, value in batch.batch.items()}
    tokenizer = CharTokenizer()

    teacher_batch, metrics = build_self_teacher_batch(batch, config(), tokenizer)
    response_width = batch.batch["responses"].shape[1]
    teacher_prompt_width = teacher_batch.batch["input_ids"].shape[1] - response_width

    assert torch.equal(
        teacher_batch.batch["input_ids"][:, teacher_prompt_width:],
        batch.batch["responses"],
    )
    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert torch.equal(
        teacher_batch.batch["response_mask"], batch.batch["response_mask"]
    )
    assert torch.equal(
        teacher_batch.batch["attention_mask"][:, teacher_prompt_width:],
        batch.batch["attention_mask"][:, -response_width:],
    )
    expected_positions = torch.clamp(
        teacher_batch.batch["attention_mask"].cumsum(dim=-1) - 1, min=0
    )
    assert torch.equal(teacher_batch.batch["position_ids"], expected_positions)
    assert metrics["distillation/feedback_trajectories"] == 2
    assert metrics["distillation/feedback_tokens"] == sum(
        len(text) for text in tokenizer.encoded_texts
    )
    assert metrics["distillation/feedback_coverage"] == pytest.approx(1.0)
    for name, value in original.items():
        assert torch.equal(batch.batch[name], value)


def test_f0_scoring_matches_same_actor_backend_on_original_causal_input():
    batch = make_batch([feedback("F0"), feedback("F0")])
    tokenizer = CharTokenizer()
    actor = TinyActorWorkerGroup()
    teacher_batch, _ = build_self_teacher_batch(batch, config(mode="F0"), tokenizer)
    direct = actor.compute_log_prob(teacher_batch).batch["old_log_probs"]
    direct = torch.where(
        batch.batch["response_mask"].bool(), direct, torch.zeros_like(direct)
    )
    actor.compute_calls = 0

    scores, metrics = score_self_teacher(batch, config(mode="F0"), tokenizer, actor)

    torch.testing.assert_close(scores, direct)
    assert actor.compute_calls == 1
    assert metrics["distillation/scored_tokens"] == 3
    assert metrics["distillation/scored_trajectories"] == 2


def test_f2_changes_causal_scores_without_mutating_student_or_disabling_adapter():
    batch = make_batch()
    tokenizer = CharTokenizer()
    actor = TinyActorWorkerGroup()
    parameter_before = actor.model.adapter_scale.detach().clone()

    scores, metrics = score_self_teacher(batch, config(), tokenizer, actor)

    assert actor.compute_calls == 1
    assert actor.meta_info_seen == [
        {"distillation_policy_version": 7, "temperature": 1.0}
    ]
    assert torch.equal(actor.model.adapter_scale.detach(), parameter_before)
    assert not scores.requires_grad
    assert scores.device.type == "cpu"
    assert torch.count_nonzero(scores[~batch.batch["response_mask"].bool()]) == 0
    assert metrics["distillation/feedback_coverage"] == pytest.approx(1.0)

    f0_batch = make_batch([feedback("F0"), feedback("F0")])
    f0_scores, _ = score_self_teacher(
        f0_batch, config(mode="F0"), CharTokenizer(), actor
    )
    assert torch.any(
        scores[batch.batch["response_mask"].bool()]
        != f0_scores[batch.batch["response_mask"].bool()]
    )


def test_scores_are_frozen_after_update_and_refreshed_for_next_batch():
    actor = TinyActorWorkerGroup()
    first, _ = score_self_teacher(
        make_batch(policy_version=7), config(), CharTokenizer(), actor
    )
    frozen = first.clone()

    with torch.no_grad():
        actor.model.adapter_scale.add_(0.7)
    assert torch.equal(first, frozen)
    second, _ = score_self_teacher(
        make_batch(policy_version=8), config(), CharTokenizer(), actor
    )

    assert actor.compute_calls == 2
    assert actor.meta_info_seen[0]["distillation_policy_version"] == 7
    assert actor.meta_info_seen[1]["distillation_policy_version"] == 8
    assert torch.any(first != second)


@pytest.mark.parametrize(
    ("feedbacks", "mode", "error"),
    [
        ([{"version": "opsd-f2-v1", "mode": "F2", "text": "x"}] * 2, "F2", "fields"),
        ([feedback("F2", text="")] * 2, "F2", "non-empty"),
        ([{**feedback(), "version": "wrong"}] * 2, "F2", "version"),
        ([feedback()] * 2, "F0", "cannot consume"),
    ],
)
def test_invalid_feedback_is_rejected(feedbacks, mode, error):
    with pytest.raises(ValueError, match=error):
        build_self_teacher_batch(
            make_batch(feedbacks), config(mode=mode), CharTokenizer()
        )


def test_missing_feedback_and_teacher_length_overflow_are_rejected():
    batch = make_batch()
    batch.non_tensor_batch.pop("opsd_feedback")
    with pytest.raises(ValueError, match="missing opsd_feedback"):
        build_self_teacher_batch(batch, config(), CharTokenizer())

    with pytest.raises(ValueError, match="feedback_max_tokens"):
        build_self_teacher_batch(
            make_batch(), config(feedback_max_tokens=4), CharTokenizer()
        )
    with pytest.raises(ValueError, match="max_teacher_length"):
        build_self_teacher_batch(
            make_batch(), config(max_teacher_length=8), CharTokenizer()
        )
    f0_batch = make_batch([feedback("F0"), feedback("F0")])
    with pytest.raises(ValueError, match="max_teacher_length"):
        build_self_teacher_batch(
            f0_batch, config(mode="F0", max_teacher_length=6), CharTokenizer()
        )


@pytest.mark.parametrize("bad_score", [0.1, float("nan")])
def test_invalid_positive_or_nan_self_teacher_scores_fail(bad_score):
    actor = CorruptActorWorkerGroup(bad_score)
    with pytest.raises(ValueError, match="invalid log-probabilities"):
        score_self_teacher(make_batch(), config(), CharTokenizer(), actor)
    assert actor.compute_calls == 1
