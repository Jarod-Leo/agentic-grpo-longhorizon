"""Feedback-conditioned scoring with the pre-update actor snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from verl.protocol import DataProto

from src.training.distillation import compact_teacher_request

_FEEDBACK_START = "[OPSD TRAINING FEEDBACK START]\n"
_FEEDBACK_END = "\n[OPSD TRAINING FEEDBACK END]\n"


def _config_value(config: Any, name: str) -> Any:
    value = (
        config.get(name, None)
        if hasattr(config, "get")
        else getattr(config, name, None)
    )
    if value is None:
        raise ValueError(f"Self-teacher config is missing {name}")
    return value


def _validate_feedback(value: Any, expected_version: str, configured_mode: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("opsd_feedback must be an object")
    required = {"version", "mode", "text", "rules"}
    if set(value) != required:
        raise ValueError(f"opsd_feedback fields must be exactly {sorted(required)}")
    if value["version"] != expected_version:
        raise ValueError(
            "opsd_feedback version does not match the configured feedback_version"
        )
    mode = value["mode"]
    text = value["text"]
    rules = value["rules"]
    if mode not in {"F0", "F2"}:
        raise ValueError("opsd_feedback mode must be F0 or F2")
    if configured_mode == "F0" and mode != "F0":
        raise ValueError("F0 scoring cannot consume F2 feedback")
    if not isinstance(text, str):
        raise ValueError("opsd_feedback text must be a string")
    if not isinstance(rules, list) or any(
        not isinstance(rule, str) or not rule for rule in rules
    ):
        raise ValueError("opsd_feedback rules must be a list of non-empty strings")
    if mode == "F0":
        if text or rules:
            raise ValueError("F0 feedback must have empty text and rules")
        return ""
    if not text.strip() or not rules:
        raise ValueError("F2 feedback must have non-empty text and rules")
    return text


def _tokenize_feedback(tokenizer: Any, text: str) -> list[int]:
    rendered = f"{_FEEDBACK_START}{text}{_FEEDBACK_END}"
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    else:
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=False)
        token_ids = encoded["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Tokenizer returned an invalid feedback batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("Tokenized F2 feedback must be non-empty")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("Tokenized feedback must contain non-negative integer IDs")
    return token_ids


def _teacher_meta_info(batch: DataProto) -> dict[str, Any]:
    meta_info = {
        key: value for key, value in batch.meta_info.items() if key != "is_lora"
    }
    meta_info["temperature"] = 1.0
    return meta_info


def build_self_teacher_batch(
    batch: DataProto, config: Any, tokenizer: Any
) -> tuple[DataProto, dict[str, float | int]]:
    """Build one actor-scoring batch without changing sampled response tokens."""
    if _config_value(config, "teacher_mode") != "self_feedback":
        raise ValueError("Self-teacher scorer requires teacher_mode=self_feedback")
    configured_mode = _config_value(config, "feedback_mode")
    if configured_mode not in {"F0", "F2"}:
        raise ValueError("feedback_mode must be F0 or F2")
    feedback_version = _config_value(config, "feedback_version")
    if not isinstance(feedback_version, str) or not feedback_version:
        raise ValueError("feedback_version must be a non-empty string")
    feedback_max_tokens = _config_value(config, "feedback_max_tokens")
    max_teacher_length = _config_value(config, "max_teacher_length")
    if (
        isinstance(feedback_max_tokens, bool)
        or not isinstance(feedback_max_tokens, int)
        or feedback_max_tokens < 1
    ):
        raise ValueError("feedback_max_tokens must be a positive integer")
    if (
        isinstance(max_teacher_length, bool)
        or not isinstance(max_teacher_length, int)
        or max_teacher_length < 2
    ):
        raise ValueError("max_teacher_length must be an integer of at least two")

    tensors = batch.batch
    required_tensors = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
    }
    missing_tensors = required_tensors.difference(tensors.keys())
    if missing_tensors:
        raise ValueError(
            f"Self-teacher batch is missing tensors: {sorted(missing_tensors)}"
        )
    if "opsd_feedback" not in batch.non_tensor_batch:
        raise ValueError("Self-teacher batch is missing opsd_feedback")
    feedback_values = batch.non_tensor_batch["opsd_feedback"]
    if len(feedback_values) != len(batch):
        raise ValueError("opsd_feedback length does not match the tensor batch")

    input_ids = tensors["input_ids"]
    attention_mask = tensors["attention_mask"]
    position_ids = tensors["position_ids"]
    responses = tensors["responses"]
    response_mask = tensors["response_mask"]
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or position_ids.shape != input_ids.shape
    ):
        raise ValueError(
            "Self-teacher input, attention and position tensors must be aligned rank-two tensors"
        )
    if (
        responses.ndim != 2
        or response_mask.shape != responses.shape
        or responses.shape[0] != input_ids.shape[0]
    ):
        raise ValueError("Self-teacher response tensors are misaligned")

    response_width = responses.shape[1]
    prompt_width = input_ids.shape[1] - response_width
    if prompt_width < 1:
        raise ValueError("Self-teacher scoring requires a non-empty prompt")

    feedback_tokens: list[list[int]] = []
    prompt_tokens: list[torch.Tensor] = []
    feedback_trajectories = 0
    for row, feedback in enumerate(feedback_values):
        compact_teacher_request(batch, row, "self-teacher")
        prompt_attention = attention_mask[row, :prompt_width].bool()
        active_prompt = prompt_attention.nonzero().flatten()
        if active_prompt.numel() == 0 or not prompt_attention[active_prompt[0] :].all():
            raise ValueError("Teacher prompt padding must be left aligned")

        text = _validate_feedback(feedback, feedback_version, configured_mode)
        prefix = _tokenize_feedback(tokenizer, text) if text else []
        if len(prefix) > feedback_max_tokens:
            raise ValueError(
                f"Feedback length {len(prefix)} exceeds feedback_max_tokens {feedback_max_tokens}"
            )
        feedback_tokens.append(prefix)
        feedback_trajectories += int(bool(prefix))
        original_prompt = input_ids[row, active_prompt[0] : prompt_width].clone()
        prefix_tensor = torch.tensor(
            prefix, dtype=input_ids.dtype, device=input_ids.device
        )
        prompt_tokens.append(torch.cat((prefix_tensor, original_prompt)))

    total_feedback_tokens = sum(len(tokens) for tokens in feedback_tokens)
    if total_feedback_tokens == 0:
        if input_ids.shape[1] > max_teacher_length:
            raise ValueError(
                f"Teacher input length {input_ids.shape[1]} exceeds max_teacher_length {max_teacher_length}"
            )
        teacher_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
                "responses": responses.clone(),
                "response_mask": response_mask.clone(),
            },
            meta_info=_teacher_meta_info(batch),
        )
    else:
        teacher_prompt_width = max(len(tokens) for tokens in prompt_tokens)
        teacher_length = teacher_prompt_width + response_width
        if teacher_length > max_teacher_length:
            raise ValueError(
                f"Teacher input length {teacher_length} exceeds max_teacher_length {max_teacher_length}"
            )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or pad_token_id < 0
        ):
            raise ValueError(
                "Tokenizer must define a non-negative integer pad_token_id"
            )

        teacher_input_ids = input_ids.new_full(
            (len(batch), teacher_length), pad_token_id
        )
        teacher_attention = attention_mask.new_zeros((len(batch), teacher_length))
        for row, prompt in enumerate(prompt_tokens):
            left_padding = teacher_prompt_width - len(prompt)
            teacher_input_ids[row, left_padding:teacher_prompt_width] = prompt
            teacher_attention[row, left_padding:teacher_prompt_width] = 1
        teacher_input_ids[:, teacher_prompt_width:] = responses
        teacher_attention[:, teacher_prompt_width:] = attention_mask[:, prompt_width:]
        teacher_positions = torch.clamp(
            teacher_attention.long().cumsum(dim=-1) - 1, min=0
        ).to(dtype=position_ids.dtype)
        teacher_batch = DataProto.from_dict(
            tensors={
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attention,
                "position_ids": teacher_positions,
                "responses": responses.clone(),
                "response_mask": response_mask.clone(),
            },
            meta_info=_teacher_meta_info(batch),
        )

    metrics: dict[str, float | int] = {
        "distillation/feedback_trajectories": feedback_trajectories,
        "distillation/feedback_tokens": total_feedback_tokens,
        "distillation/feedback_coverage": feedback_trajectories / len(batch),
    }
    return teacher_batch, metrics


def score_self_teacher(
    batch: DataProto, config: Any, tokenizer: Any, actor_wg: Any
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Score the enhanced batch once with the current adapter-enabled actor."""
    if actor_wg is None or not hasattr(actor_wg, "compute_log_prob"):
        raise ValueError("Self-teacher scoring requires an actor worker group")
    teacher_batch, metrics = build_self_teacher_batch(batch, config, tokenizer)
    output = actor_wg.compute_log_prob(teacher_batch)
    if not hasattr(output, "batch") or "old_log_probs" not in output.batch:
        raise ValueError("Self-teacher actor did not return old_log_probs")

    scores = output.batch["old_log_probs"]
    response_mask = batch.batch["response_mask"].bool()
    if scores.shape != response_mask.shape:
        raise ValueError("Self-teacher log-probabilities do not align with responses")
    if not torch.isfinite(scores).all() or torch.any(scores[response_mask] > 1e-6):
        raise ValueError("Self-teacher returned invalid log-probabilities")
    scores = (
        torch.where(response_mask, scores, torch.zeros_like(scores))
        .detach()
        .clone()
        .cpu()
    )

    metrics.update(
        {
            "distillation/scored_tokens": int(response_mask.sum()),
            "distillation/scored_trajectories": len(batch),
        }
    )
    return scores, metrics


__all__ = ["build_self_teacher_batch", "score_self_teacher"]
