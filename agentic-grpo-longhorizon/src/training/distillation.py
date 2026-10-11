"""Score a fixed student batch before its first optimizer step."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from numbers import Integral
from pathlib import Path
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
    return _score_teacher(batch, config, tokenizer)


def _teacher_identity(config: Any) -> dict:
    identity = {
        "model_id": config.teacher_model_id,
        "revision": config.teacher_revision,
    }
    if getattr(config, "teacher_verify_revision", False):
        identity["revision_verified"] = True
    path = getattr(config, "teacher_adapter_path", None)
    digest = getattr(config, "teacher_adapter_sha256", None)
    if bool(path) != bool(digest):
        raise ValueError("Teacher adapter path and SHA256 must be supplied together")
    if path:
        identity.update(adapter_path=str(Path(path).resolve()), adapter_sha256=digest)
    return identity


def _score_teacher(
    batch: Any,
    config: Any,
    tokenizer: Any,
    prepare_request: Callable[[dict, int], dict] | None = None,
) -> tuple[torch.Tensor, dict]:
    from src.models.token_teacher import input_ids_sha256, tokenizer_fingerprint

    fingerprint = tokenizer_fingerprint(tokenizer)
    identity = _teacher_identity(config)
    scores = torch.zeros_like(batch.batch["old_log_probs"], device="cpu")
    active_tokens = 0
    for row in range(len(batch)):
        request, positions = compact_teacher_request(batch, row, fingerprint)
        if prepare_request is not None:
            request = prepare_request(request, row)
        request.update(identity)
        result = post_json(
            config.endpoint.rstrip("/") + "/score", request, float(config.timeout_s)
        )
        expected = {
            "input_sha256": input_ids_sha256(request["input_ids"]),
            "response_start": request["response_start"],
            "response_mask": request["response_mask"],
            "tokenizer_fingerprint": fingerprint,
            **identity,
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


@lru_cache(maxsize=8)
def _training_task_ids(split_path: str) -> frozenset[int]:
    split = json.loads(Path(split_path).read_text())
    train = frozenset(split["seen_task_ids"])
    test = frozenset(split["unseen_task_ids"])
    if train & test:
        raise ValueError("Train/test task IDs overlap")
    return train


@lru_cache(maxsize=8)
def _fixed_skill_bank(path: str, digest: str, split_path: str) -> dict[int, dict]:
    from src.training.skill_bank import load_skill_bank

    return load_skill_bank(path, digest, split_path)


def score_skill_teacher(
    batch: Any, config: Any, tokenizer: Any, actor_wg: Any = None
) -> tuple[torch.Tensor, dict]:
    """Score the student trajectory after adding an offline skill to teacher history.

    Prefix tokens are encoded independently; sampled history and response IDs
    are never decoded or tokenized again. Validation and held-out tasks cannot
    enter this training-only scorer, including the no-skill ablation.
    """
    from src.training.skill_bank import render_skill

    if batch.meta_info.get("validate", False):
        raise ValueError("Skill teacher scorer cannot score validation batches")
    split_path = str(Path(config.split_path).resolve())
    train_ids = _training_task_ids(split_path)
    extra_infos = batch.non_tensor_batch.get("extra_info")
    if extra_infos is None or len(extra_infos) != len(batch):
        raise ValueError("Skill teacher requires per-row extra_info")
    task_ids = []
    for extra in extra_infos:
        if not isinstance(extra, dict) or extra.get("split") != "train":
            raise ValueError("Skill teacher only accepts split=train")
        task_id = extra.get("task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, Integral)
            or task_id not in train_ids
        ):
            raise ValueError("Skill teacher requires a training task_id")
        nested = extra.get("interaction_kwargs", {}).get("task_id", task_id)
        if nested != task_id:
            raise ValueError("Task ID differs from interaction_kwargs.task_id")
        task_ids.append(int(task_id))

    use_skills = bool(config.use_skills)
    bank = (
        _fixed_skill_bank(
            str(Path(config.skill_bank_path).resolve()),
            str(config.skill_bank_sha256),
            split_path,
        )
        if use_skills
        else {}
    )
    max_prefix = int(config.skill_prefix_max_tokens)
    max_length = int(config.teacher_max_length)
    if not 0 < max_prefix <= 512:
        raise ValueError("skill_prefix_max_tokens must be between 1 and 512")
    prefixes = {}
    for task_id in set(task_ids):
        if task_id not in bank:
            continue
        prefix = tokenizer.encode(render_skill(bank[task_id]), add_special_tokens=False)
        if not prefix or len(prefix) > max_prefix:
            raise ValueError(
                f"Task {task_id} skill prefix has {len(prefix)} tokens; limit is {max_prefix}"
            )
        prefixes[task_id] = prefix

    def prepare(request: dict, row: int) -> dict:
        prefix = prefixes.get(task_ids[row], [])
        request["input_ids"] = prefix + request["input_ids"]
        request["response_start"] += len(prefix)
        if len(request["input_ids"]) > max_length:
            raise ValueError(f"Skill teacher input exceeds max_length {max_length}")
        return request

    scores, metrics = _score_teacher(batch, config, tokenizer, prepare)
    covered = sum(task_id in prefixes for task_id in task_ids)
    metrics.update(
        {
            "distillation/skill_trajectories": covered,
            "distillation/skill_fallback_trajectories": len(batch) - covered,
            "distillation/skill_coverage": covered / len(batch) if len(batch) else 0.0,
            "distillation/skill_tokens": sum(
                len(prefixes.get(task_id, [])) for task_id in task_ids
            ),
        }
    )
    return scores, metrics
