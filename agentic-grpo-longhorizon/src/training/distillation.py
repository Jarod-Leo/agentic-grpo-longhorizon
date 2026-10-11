"""Score a fixed student batch before its first optimizer step."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from numbers import Integral, Real
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
def _fixed_skill_bank(path: str, digest: str, split_path: str) -> dict[int, list[dict]]:
    from src.training.skill_bank import load_skill_bank

    return load_skill_bank(path, digest, split_path)


def _batch_training_task_ids(batch: Any, config: Any) -> list[int]:
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

    return task_ids


class SkillTeacherState:
    """Bind pre-rollout task-local selection to rewards and checkpoint state."""

    def __init__(self, config: Any) -> None:
        from src.training.skill_bank import file_sha256, load_skill_bank
        from src.training.skill_selector import TaskSkillSelector

        self.config = config
        self.bank = load_skill_bank(
            config.skill_bank_path, config.skill_bank_sha256, config.split_path
        )
        self.selector = TaskSkillSelector(
            self.bank,
            bank_sha256=config.skill_bank_sha256,
            split_sha256=file_sha256(config.split_path),
            exploration_coef=float(config.ucb_exploration_coef),
            mode=str(config.skill_selection),
        )
        log_path = getattr(config, "selection_log_path", None)
        if not log_path:
            if not os.environ.get("RUN_DIR"):
                raise ValueError(
                    "Set selection_log_path or RUN_DIR for the skill selection audit"
                )
            log_path = str(Path(os.environ["RUN_DIR"]) / "skill_selection.jsonl")
        self.audit_path = Path(log_path)
        self.prepared_step: int | None = None
        self.selected: dict[int, dict | None] = {}

    def prepare(self, batch: Any, step: int) -> None:
        task_ids = _batch_training_task_ids(batch, self.config)
        self.selected = self.selector.select(task_ids, step)
        self.prepared_step = step

    def selected_for(self, task_id: int, step: int) -> dict | None:
        if step != self.prepared_step or task_id not in self.selected:
            raise ValueError(
                "Skill teacher selection was not prepared before this rollout step"
            )
        return self.selected[task_id]

    def commit(self, batch: Any, step: int) -> dict:
        task_ids = _batch_training_task_ids(batch, self.config)
        if step != self.prepared_step or set(task_ids) != set(self.selected):
            raise ValueError(
                "Outcome batch does not match the selected task group and step"
            )
        outcomes = batch.non_tensor_batch.get("outcome_reward")
        if outcomes is None or len(outcomes) != len(task_ids):
            raise ValueError(
                "Skill selection requires original per-trajectory outcome_reward"
            )
        rewards: dict[int, list[float]] = {}
        for task_id, outcome in zip(task_ids, outcomes, strict=True):
            if not isinstance(outcome, Real) or outcome not in (0, 1):
                raise ValueError("Skill selection outcome_reward must be binary 0 or 1")
            rewards.setdefault(task_id, []).append(float(outcome))
        records = self.selector.commit(rewards, step)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        {
                            **record,
                            "global_step": step,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        self.prepared_step = None
        self.selected = {}
        assigned = [row for row in records if row["skill_id"] is not None]
        count = len(assigned)
        prior_scores = [
            row["score_before"] for row in assigned if row["score_before"] is not None
        ]
        return {
            "distillation/skill_reward_groups": count,
            "distillation/skill_reward_mean": sum(
                row["reward_mean"] for row in assigned
            )
            / count
            if count
            else 0.0,
            "distillation/skill_selection_count_before": sum(
                row["count_before"] for row in assigned
            )
            / count
            if count
            else 0.0,
            "distillation/skill_selection_count_after": sum(
                row["count_after"] for row in assigned
            )
            / count
            if count
            else 0.0,
            "distillation/skill_selection_score_before": sum(prior_scores)
            / len(prior_scores)
            if prior_scores
            else 0.0,
            "distillation/skill_selection_reward_before": sum(
                row["mean_reward_before"] for row in assigned
            )
            / count
            if count
            else 0.0,
        }

    def save_checkpoint(self, checkpoint_dir: str | Path, step: int) -> None:
        state = self.selector.state_dict()
        if self.selector.last_step != step:
            raise ValueError("Skill selector state does not match the saved model step")
        path = Path(checkpoint_dir) / "skill_selector.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": "skill-selector-checkpoint-v1",
                    "global_step": step,
                    "selector": state,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.replace(path)

    def load_checkpoint(self, checkpoint_dir: str | Path, step: int) -> None:
        state = json.loads((Path(checkpoint_dir) / "skill_selector.json").read_text())
        if (
            state.get("version") != "skill-selector-checkpoint-v1"
            or state.get("global_step") != step
        ):
            raise ValueError("Skill selector checkpoint version or model step mismatch")
        self.selector.load_state_dict(state["selector"])
        if self.selector.last_step != step:
            raise ValueError("Skill selector restored step differs from the model step")
        self.prepared_step = None
        self.selected = {}


def score_skill_teacher(
    batch: Any,
    config: Any,
    tokenizer: Any,
    actor_wg: Any = None,
    state_manager: SkillTeacherState | None = None,
) -> tuple[torch.Tensor, dict]:
    """Score untouched student tokens with the skill selected before rollout."""
    from src.training.skill_bank import render_skill

    task_ids = _batch_training_task_ids(batch, config)
    skills: dict[int, dict | None] = {}
    if bool(config.use_skills):
        if (
            getattr(config, "skill_selection", "first") == "ucb"
            and state_manager is None
        ):
            raise ValueError(
                "UCB skill scoring requires a pre-rollout state_manager selection"
            )
        if state_manager is not None:
            step = int(batch.meta_info["distillation_policy_version"]) + 1
            skills = {
                task_id: state_manager.selected_for(task_id, step)
                for task_id in set(task_ids)
            }
        else:
            bank = _fixed_skill_bank(
                str(Path(config.skill_bank_path).resolve()),
                str(config.skill_bank_sha256),
                str(Path(config.split_path).resolve()),
            )
            for task_id in set(task_ids):
                candidates = bank.get(task_id, [])
                if len(candidates) > 1:
                    raise ValueError(
                        "Multiple task skills require a pre-rollout state_manager selection"
                    )
                skills[task_id] = candidates[0] if candidates else None
    max_prefix = int(config.skill_prefix_max_tokens)
    max_length = int(config.teacher_max_length)
    if not 0 < max_prefix <= 512:
        raise ValueError("skill_prefix_max_tokens must be between 1 and 512")
    prefixes = {}
    for task_id in set(task_ids):
        if skills.get(task_id) is None:
            continue
        prefix = tokenizer.encode(
            render_skill(skills[task_id]), add_special_tokens=False
        )
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
