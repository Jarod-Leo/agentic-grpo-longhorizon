"""Frozen causal language-model teacher used by online policy distillation.

The teacher consumes the exact token IDs produced by the policy rollout.  It
does not render chat templates, tokenize text, or generate replacement
answers.  This keeps OPD aligned with the trajectory that was actually
sampled.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class TokenTeacherValidationError(ValueError):
    """Raised when a scoring request violates the token-teacher protocol."""


def _json_value(value: Any) -> Any:
    """Convert tokenizer metadata to a deterministic JSON value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "content"):
        return str(value.content)
    return str(value)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash token-to-ID semantics, special tokens, and the chat template."""

    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise TokenTeacherValidationError(
            "tokenizer.get_vocab() must return a non-empty mapping"
        )

    normalized_vocab = sorted(
        (str(token), int(token_id)) for token, token_id in vocab.items()
    )
    payload = {
        "vocab": normalized_vocab,
        "special_tokens_map": _json_value(getattr(tokenizer, "special_tokens_map", {})),
        "chat_template": _json_value(getattr(tokenizer, "chat_template", None)),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_ids_sha256(input_ids: Sequence[int]) -> str:
    """Return the protocol hash of an exact token-ID sequence."""

    encoded = json.dumps(
        list(input_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FrozenTokenTeacher:
    """Score sampled response tokens with a frozen causal language model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_id: str,
        revision: str,
        max_length: int = 24576,
        logit_chunk_size: int = 128,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if not revision:
            raise ValueError("revision must be non-empty")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.logit_chunk_size = logit_chunk_size
        self.tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)

        vocab = tokenizer.get_vocab()
        vocab_ids = {int(token_id) for token_id in vocab.values()}
        special_ids = {
            int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])
        }
        self._token_ids = frozenset(vocab_ids | special_ids)

        self.model.requires_grad_(False)
        self.model.eval()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "max_length": self.max_length,
        }

    def score(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return teacher log probabilities at assistant-token positions.

        ``response_mask`` covers the complete response suffix beginning at
        ``response_start``.  Its zero entries are returned as exact ``0.0``.
        Logits at position ``t - 1`` score the token at position ``t``.
        """

        input_ids, response_start, response_mask, policy_version = (
            self._validate_request(request)
        )
        response_length = len(response_mask)
        input_hash = input_ids_sha256(input_ids)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        token_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(
                input_ids=token_tensor, use_cache=False, return_dict=True
            )
            logits = output.logits

        if (
            logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] != len(input_ids)
        ):
            raise RuntimeError(
                "teacher returned invalid logits shape: "
                f"expected [1, {len(input_ids)}, vocab], got {tuple(logits.shape)}"
            )
        if logits.shape[-1] <= max(input_ids):
            raise TokenTeacherValidationError(
                f"token ID {max(input_ids)} exceeds teacher logits vocabulary {logits.shape[-1]}"
            )

        shifted_logits = logits[0, response_start - 1 : len(input_ids) - 1]
        targets = token_tensor[0, response_start:]
        mask = torch.tensor(response_mask, dtype=torch.bool, device=logits.device)
        log_probs = torch.zeros(
            response_length, dtype=torch.float32, device=logits.device
        )

        for offset in range(0, response_length, self.logit_chunk_size):
            end = min(offset + self.logit_chunk_size, response_length)
            block = shifted_logits[offset:end]
            if not torch.isfinite(block).all():
                raise RuntimeError(
                    f"teacher produced non-finite logits in response positions {offset}:{end}"
                )

            active = mask[offset:end]
            if not active.any():
                continue
            active_logits = block[active].float()
            active_targets = targets[offset:end][active].to(active_logits.device)
            selected = active_logits.gather(
                dim=-1, index=active_targets.unsqueeze(-1)
            ).squeeze(-1)
            active_log_probs = selected - torch.logsumexp(active_logits, dim=-1)
            if not torch.isfinite(active_log_probs).all():
                raise RuntimeError(
                    f"teacher produced non-finite log probabilities in response positions {offset}:{end}"
                )
            log_probs[offset:end][active] = active_log_probs

        teacher_log_probs = log_probs.cpu().tolist()
        if len(teacher_log_probs) != response_length or not all(
            math.isfinite(value) for value in teacher_log_probs
        ):
            raise RuntimeError("teacher returned invalid log probabilities")
        if any(
            teacher_log_probs[index] != 0.0
            for index, enabled in enumerate(response_mask)
            if not enabled
        ):
            raise RuntimeError("masked teacher log probabilities must be zero")

        return {
            "teacher_log_probs": teacher_log_probs,
            "input_sha256": input_hash,
            "response_start": response_start,
            "response_mask": response_mask,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "model_id": self.model_id,
            "revision": self.revision,
            "policy_version": policy_version,
        }

    def _validate_request(
        self, request: Mapping[str, Any]
    ) -> tuple[list[int], int, list[int], int]:
        if not isinstance(request, Mapping):
            raise TokenTeacherValidationError("request must be a JSON object")

        required = {
            "input_ids",
            "response_start",
            "response_mask",
            "tokenizer_fingerprint",
            "policy_version",
        }
        missing = required.difference(request)
        if missing:
            raise TokenTeacherValidationError(
                f"request is missing required fields: {sorted(missing)}"
            )

        raw_ids = request["input_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise TokenTeacherValidationError("input_ids must be a non-empty list")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in raw_ids
        ):
            raise TokenTeacherValidationError("input_ids must contain integers")
        input_ids = list(raw_ids)
        if len(input_ids) > self.max_length:
            raise TokenTeacherValidationError(
                f"input length {len(input_ids)} exceeds teacher max_length {self.max_length}"
            )
        invalid_ids = [
            token_id
            for token_id in input_ids
            if token_id < 0 or token_id not in self._token_ids
        ]
        if invalid_ids:
            raise TokenTeacherValidationError(
                f"input_ids contain IDs absent from the tokenizer vocabulary: {invalid_ids[:4]}"
            )

        response_start = request["response_start"]
        if isinstance(response_start, bool) or not isinstance(response_start, int):
            raise TokenTeacherValidationError("response_start must be an integer")
        if not 1 <= response_start < len(input_ids):
            raise TokenTeacherValidationError(
                "response_start must satisfy 1 <= response_start < len(input_ids)"
            )

        raw_mask = request["response_mask"]
        if not isinstance(raw_mask, list):
            raise TokenTeacherValidationError("response_mask must be a list")
        if len(raw_mask) != len(input_ids) - response_start:
            raise TokenTeacherValidationError(
                "response_mask length must equal len(input_ids) - response_start"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in raw_mask
        ):
            raise TokenTeacherValidationError(
                "response_mask must contain only integer 0 or 1 values"
            )
        response_mask = list(raw_mask)
        if not any(response_mask):
            raise TokenTeacherValidationError(
                "response_mask must contain at least one assistant token"
            )

        fingerprint = request["tokenizer_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or fingerprint != self.tokenizer_fingerprint
        ):
            raise TokenTeacherValidationError(
                "tokenizer_fingerprint does not match the teacher tokenizer"
            )

        policy_version = request["policy_version"]
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise TokenTeacherValidationError(
                "policy_version must be a non-negative integer"
            )

        return input_ids, response_start, response_mask, policy_version
