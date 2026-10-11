"""Task-local UCB selection using only committed training outcome groups."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

SELECTOR_VERSION = "task-skill-selector-v1"


class TaskSkillSelector:
    """Choose once per task and update once after a successful training step.

    Selection is transactional: retries return the pending choices, and only
    ``commit`` changes counts. Checkpoints must have no pending selection.
    """

    def __init__(
        self,
        bank: dict[int, list[dict]],
        bank_sha256: str,
        split_sha256: str,
        exploration_coef: float = math.sqrt(2),
        mode: str = "ucb",
    ) -> None:
        if mode not in ("ucb", "first"):
            raise ValueError("Skill selection mode must be ucb or first")
        if not math.isfinite(exploration_coef) or exploration_coef < 0:
            raise ValueError("Exploration coefficient must be finite and nonnegative")
        if not bank_sha256 or not split_sha256:
            raise ValueError("Bank and split hashes are required")
        self.bank = {task_id: list(skills) for task_id, skills in bank.items()}
        skill_ids = set()
        for task_id, skills in self.bank.items():
            if type(task_id) is not int:
                raise ValueError("Skill bank task IDs must be integers")
            for skill in skills:
                skill_id = skill.get("skill_id")
                if skill.get("task_id") != task_id:
                    raise ValueError("Skill must belong to its task")
                if not isinstance(skill_id, str) or not skill_id or skill_id in skill_ids:
                    raise ValueError("Skill IDs must be nonempty and globally unique")
                skill_ids.add(skill_id)
        self.bank_sha256 = bank_sha256
        self.split_sha256 = split_sha256
        self.exploration_coef = float(exploration_coef)
        self.mode = mode
        self.last_step: int | None = None
        self._statistics = {
            task_id: {skill["skill_id"]: {"n": 0, "sum_reward": 0.0} for skill in skills}
            for task_id, skills in self.bank.items()
        }
        self._pending: dict[int, dict[str, Any]] | None = None
        self._pending_step: int | None = None

    def select(self, task_ids: Iterable[int], step: int) -> dict[int, dict | None]:
        """Select one candidate per distinct task without changing statistics."""
        tasks = list(dict.fromkeys(task_ids))
        if type(step) is not int or step < 0 or any(type(task_id) is not int for task_id in tasks):
            raise ValueError("Selection requires integer task IDs and a nonnegative step")
        if self._pending is not None:
            if step != self._pending_step or set(tasks) != set(self._pending):
                raise ValueError("Cannot overwrite a pending skill selection")
            return {task_id: pending["skill"] for task_id, pending in self._pending.items()}
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("Selection step must follow the last committed step")

        pending = {}
        for task_id in tasks:
            skills = self.bank.get(task_id, [])
            statistics = self._statistics.get(task_id, {})
            total = sum(stat["n"] for stat in statistics.values())
            skill = None
            score = None
            count = 0
            mean = 0.0
            if skills:
                if self.mode == "first":
                    skill = skills[0]
                else:
                    skill = next(
                        (candidate for candidate in skills if statistics[candidate["skill_id"]]["n"] == 0), None
                    )
                    if skill is None:
                        scores = []
                        for candidate in skills:
                            stat = statistics[candidate["skill_id"]]
                            candidate_score = stat["sum_reward"] / stat["n"] + self.exploration_coef * math.sqrt(
                                math.log(total) / stat["n"]
                            )
                            scores.append((candidate_score, candidate))
                        # max keeps the first candidate on equal scores.
                        score, skill = max(scores, key=lambda entry: entry[0])
                stat = statistics[skill["skill_id"]]
                count = stat["n"]
                mean = stat["sum_reward"] / count if count else 0.0
                if self.mode == "first":
                    score = mean
            pending[task_id] = {
                "skill": skill,
                "score_before": score,  # null denotes an untried candidate or no candidate.
                "count_before": count,
                "mean_reward_before": mean,
                "task_count_before": total,
            }
        self._pending = pending
        self._pending_step = step
        return {task_id: choice["skill"] for task_id, choice in pending.items()}

    def commit(self, task_rewards: dict[int, list[float]], step: int) -> list[dict]:
        """Commit each task's mean binary outcome as one candidate observation."""
        if type(step) is not int or self._pending is None or step != self._pending_step:
            raise ValueError("Commit must match a pending selection step")
        if set(task_rewards) != set(self._pending):
            raise ValueError("Commit rewards must match the selected task set")
        reward_means = {}
        # Validate every group first so a bad group cannot partially update counts.
        for task_id, outcomes in task_rewards.items():
            if not outcomes or any(outcome not in (0, 1) for outcome in outcomes):
                raise ValueError("Each task requires a nonempty list of binary outcomes")
            reward_means[task_id] = sum(outcomes) / len(outcomes)

        audit = []
        for task_id, choice in self._pending.items():
            skill = choice["skill"]
            count_after = 0
            sum_reward_after = 0.0
            task_count_after = choice["task_count_before"]
            if skill is not None:
                stat = self._statistics[task_id][skill["skill_id"]]
                stat["n"] += 1
                stat["sum_reward"] += reward_means[task_id]
                count_after = stat["n"]
                sum_reward_after = stat["sum_reward"]
                task_count_after += 1
            audit.append(
                {
                    "step": step,
                    "task_id": task_id,
                    "skill_id": skill["skill_id"] if skill is not None else None,
                    "mode": self.mode,
                    "score_before": choice["score_before"],
                    "mean_reward_before": choice["mean_reward_before"],
                    "count_before": choice["count_before"],
                    "task_count_before": choice["task_count_before"],
                    "reward_mean": reward_means[task_id],
                    "reward_count": len(task_rewards[task_id]),
                    "count_after": count_after,
                    "sum_reward_after": sum_reward_after,
                    "mean_reward_after": sum_reward_after / count_after if count_after else 0.0,
                    "task_count_after": task_count_after,
                }
            )
        self.last_step = step
        self._pending = None
        self._pending_step = None
        return audit

    def state_dict(self) -> dict:
        """Return a JSON-ready committed state; never checkpoint pending choices."""
        if self._pending is not None:
            raise ValueError("Cannot checkpoint a pending skill selection")
        return {
            "version": SELECTOR_VERSION,
            "bank_sha256": self.bank_sha256,
            "split_sha256": self.split_sha256,
            "exploration_coef": self.exploration_coef,
            "mode": self.mode,
            "last_step": self.last_step,
            "statistics": {
                str(task_id): {skill_id: dict(stat) for skill_id, stat in statistics.items()}
                for task_id, statistics in self._statistics.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore counts after validating bank, split and selector configuration."""
        if self._pending is not None:
            raise ValueError("Cannot restore over a pending skill selection")
        for name, expected in (
            ("version", SELECTOR_VERSION),
            ("bank_sha256", self.bank_sha256),
            ("split_sha256", self.split_sha256),
            ("exploration_coef", self.exploration_coef),
            ("mode", self.mode),
        ):
            if state.get(name) != expected:
                raise ValueError(f"Skill selector {name} mismatch")
        last_step = state["last_step"]
        if last_step is not None and (type(last_step) is not int or last_step < 0):
            raise ValueError("Invalid skill selector last_step")
        saved = state["statistics"]
        if set(saved) != {str(task_id) for task_id in self._statistics}:
            raise ValueError("Skill selector task statistics mismatch")
        restored = {}
        for task_id, statistics in self._statistics.items():
            task_stats = saved[str(task_id)]
            if set(task_stats) != set(statistics):
                raise ValueError("Skill selector candidate statistics mismatch")
            restored[task_id] = {}
            for skill_id, stat in task_stats.items():
                count, reward_sum = stat["n"], stat["sum_reward"]
                if type(count) is not int or count < 0 or not isinstance(reward_sum, (int, float)):
                    raise ValueError("Invalid skill selector statistics")
                if not math.isfinite(reward_sum) or not 0 <= reward_sum <= count:
                    raise ValueError("Invalid skill selector reward sum")
                restored[task_id][skill_id] = {"n": count, "sum_reward": float(reward_sum)}
        self._statistics = restored
        self.last_step = last_step
