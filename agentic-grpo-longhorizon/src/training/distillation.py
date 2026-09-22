"""Score a fixed student batch before its first optimizer step."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any

import torch


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def compact_teacher_request(
    batch: Any, row: int, fingerprint: str
) -> tuple[dict, torch.Tensor]:
    """Remove padding only; keep all real user/tool history and original token IDs."""
    tensors = batch.batch
    ids = tensors["input_ids"][row].cpu()
    attention = tensors["attention_mask"][row].cpu().bool()
    mask = tensors["response_mask"][row].cpu()
    response_width = tensors["responses"].shape[1]
    prompt_width = len(ids) - response_width
    if prompt_width <= 0 or not torch.equal(
        ids[prompt_width:], tensors["responses"][row].cpu()
    ):
        raise ValueError("Response IDs are not the input suffix")
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(
        mask.bool() & ~attention[prompt_width:]
    ):
        raise ValueError("Invalid response mask or active padding")
    response_positions = attention[prompt_width:].nonzero().flatten()
    compact_mask = mask[response_positions].to(torch.long).tolist()
    if not any(compact_mask):
        raise ValueError("Teacher batch has no assistant tokens")
    request = {
        "input_ids": ids[attention].tolist(),
        "response_start": int(attention[:prompt_width].sum()),
        "response_mask": compact_mask,
        "tokenizer_fingerprint": fingerprint,
        "policy_version": int(batch.meta_info["distillation_policy_version"]),
    }
    if request["response_start"] < 1:
        raise ValueError("Teacher scoring requires a causal prompt")
    return request, response_positions


def score_external_teacher(
    batch: Any, config: Any, tokenizer: Any, actor_wg: Any = None
) -> tuple[torch.Tensor, dict]:
    """The external model scores the exact sampled sequence, at temperature one."""
    from src.models.token_teacher import tokenizer_fingerprint

    fingerprint = tokenizer_fingerprint(tokenizer)
    scores = torch.zeros_like(batch.batch["old_log_probs"], device="cpu")
    active_tokens = 0
    for row in range(len(batch)):
        request, positions = compact_teacher_request(batch, row, fingerprint)
        result = post_json(
            config.endpoint.rstrip("/") + "/score", request, float(config.timeout_s)
        )
        digest = hashlib.sha256(
            json.dumps(request["input_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        expected = {
            "input_sha256": digest,
            "response_start": request["response_start"],
            "response_mask": request["response_mask"],
            "tokenizer_fingerprint": fingerprint,
            "model_id": config.teacher_model_id,
            "revision": config.teacher_revision,
            "policy_version": request["policy_version"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Teacher identity, token alignment or policy version mismatch"
            )
        values = torch.tensor(result["teacher_log_probs"], dtype=scores.dtype)
        active = torch.tensor(request["response_mask"], dtype=torch.bool)
        if values.shape != active.shape or not torch.isfinite(values).all():
            raise ValueError("Teacher log-prob shape or finite-value check failed")
        if torch.any(values[~active] != 0) or torch.any(values[active] > 1e-6):
            raise ValueError(
                "Teacher returned invalid probabilities or nonzero masked scores"
            )
        scores[row, positions] = values
        active_tokens += int(active.sum())
    return scores, {
        "distillation/scored_tokens": active_tokens,
        "distillation/scored_trajectories": len(batch),
    }
