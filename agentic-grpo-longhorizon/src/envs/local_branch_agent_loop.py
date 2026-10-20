"""Frozen-policy local branching diagnostics for tau-bench evaluation."""

from __future__ import annotations

import math
import time
from typing import Any

from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from src.envs.tau_bench_context import CURRENT_TAU_STATE
from src.envs.tau_bench_interaction import record_assistant_content
from src.evaluation.local_branching import (
    append_branch_group,
    clone_agent_data,
    leave_one_out_advantages,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState


class LocalBranchDiagnosticAgentLoop(TauBenchAgentLoop):
    """Run K x M suffix branches after an unchanged validation backbone."""

    def _branch_setting(self, name: str, default: Any) -> Any:
        config = getattr(self, "_branch_config", None)
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(name, default)
        return getattr(config, name, default)

    def _local_branch_config(self):
        return self.config.actor_rollout_ref.rollout.multi_turn.local_branch

    def _branch_output_dir(self, config) -> str:
        output_dir = config.get("output_dir")
        if output_dir:
            return str(output_dir)
        run_dir = self.config.get("trainer", {}).get("validation_data_dir")
        if not run_dir:
            raise ValueError("local_branch.output_dir is required")
        return f"{run_dir}/local_branches"

    def _skip_record(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": "local_branch_root_action_v1",
            "record_type": "group",
            "root_trajectory_id": getattr(self.eval_data, "request_id", None),
            "task_id": self._branch_task_id,
            "split": self._branch_extra_info.get("split"),
            "step": self._branch_trajectory.get("step"),
            "sample_index": self._branch_trajectory.get("sample_index"),
            "rollout_n": self._branch_trajectory.get("rollout_n"),
            "trigger": self._branch_trigger,
            "candidates_per_prefix": int(self._branch_config.candidates),
            "continuations_per_candidate": int(self._branch_config.continuations),
            "skipped": True,
            "skip_reason": reason,
            "root_action_span": None,
            "root_action_token_ids": None,
            "root_local_advantage": None,
            "effective": None,
            "candidates": [],
            "cost": None,
            "auxiliary_trajectory_ids": [],
        }

    def _critical_prefix_trigger(self, data) -> str | None:
        get_state = getattr(data.interaction, "opsd_feedback_state", None)
        state = (
            get_state(data.request_id)
            if get_state is not None
            else CURRENT_TAU_STATE.get()
        )
        history = state.get("action_history", []) if state else []
        if not history:
            return None
        previous = history[-1]
        if previous.get("is_error"):
            return "after_tool_error"
        tool = previous.get("tool")
        if tool in set(self._branch_setting("trigger_tools", [])):
            return f"after_successful_tool:{tool}"
        return None

    def _eligible_anchor(self, data) -> str | None:
        state = CURRENT_TAU_STATE.get() or {}
        if state.get("contaminated"):
            self._branch_skip_reason = "contaminated_prefix"
            return None
        if state.get("done"):
            self._branch_skip_reason = "terminal_before_anchor"
            return None
        if self._branch_snapshot is not None or getattr(
            self, "_branch_attempted", False
        ):
            return None
        if self._branch_setting("trigger", "anchor_turn") == "critical_prefix":
            trigger = self._critical_prefix_trigger(data)
        else:
            trigger = (
                f"assistant_turn:{self._branch_anchor_turn}"
                if data.assistant_turns + 1 == self._branch_anchor_turn
                else None
            )
        if trigger is None:
            return None
        self._branch_attempted = True
        self._branch_trigger = trigger
        if (
            self.max_assistant_turns
            and data.assistant_turns + 1 >= self.max_assistant_turns
        ):
            self._branch_skip_reason = "root_action_exhausts_assistant_budget"
            return None
        if self.max_user_turns and data.user_turns + 1 >= self.max_user_turns:
            self._branch_skip_reason = "root_action_exhausts_user_budget"
            return None
        if len(data.response_mask) >= self.response_length:
            self._branch_skip_reason = "root_action_exhausts_response_budget"
            return None
        return trigger

    async def _handle_generating_state(
        self, data, sampling_params, ignore_termination=False
    ):
        trigger = self._eligible_anchor(data) if self._branch_enabled else None
        captured = None
        if trigger is not None:
            captured = {
                "agent_data": clone_agent_data(data, data.request_id),
                "interaction": await data.interaction.snapshot_interaction(
                    data.request_id
                ),
                "trigger": trigger,
                "prefix": {
                    "assistant_turns": data.assistant_turns,
                    "user_turns": data.user_turns,
                    "response_tokens": len(data.response_mask),
                    "policy_tokens": sum(data.response_mask),
                    "tool_calls": data.total_tool_calls,
                    "message_count": len(data.messages),
                },
            }
        state = await super()._handle_generating_state(
            data, sampling_params, ignore_termination
        )
        if captured is not None:
            action_tokens = list(data.response_ids)
            action_start = captured["prefix"]["response_tokens"]
            action_end = action_start + len(action_tokens)
            if state == AgentState.TERMINATED or not action_tokens:
                self._branch_skip_reason = "root_action_not_executable"
            elif data.response_mask[action_start:action_end] != [1] * len(
                action_tokens
            ):
                self._branch_skip_reason = "root_action_mask_mismatch"
            else:
                captured.update(
                    {
                        "root_candidate_data": clone_agent_data(
                            data, data.request_id
                        ),
                        "root_candidate_state": state,
                        "root_action_token_ids": action_tokens,
                        "root_action_span": [action_start, action_end],
                    }
                )
                self._branch_snapshot = captured
        return state

    @staticmethod
    def _outcome_score(raw_score: Any) -> float:
        if isinstance(raw_score, dict):
            if "outcome_score" not in raw_score:
                raise ValueError("Branch score is missing outcome_score")
            raw_score = raw_score["outcome_score"]
        score = float(raw_score)
        if not math.isfinite(score) or score not in (0.0, 1.0):
            raise ValueError(f"Branch outcome_score must be binary, got {score!r}")
        return score

    async def _continue_branch(
        self, data, initial_state: AgentState, sampling_params: dict[str, Any]
    ):
        state = initial_state
        termination = "unknown"
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = await TauBenchAgentLoop._handle_generating_state(
                    self, data, sampling_params
                )
                if state == AgentState.TERMINATED:
                    termination = self.eval_termination
            elif state == AgentState.PROCESSING_TOOLS:
                state = await TauBenchAgentLoop._handle_processing_tools_state(self, data)
                if state == AgentState.TERMINATED:
                    termination = self.eval_termination
            elif state == AgentState.INTERACTING:
                state = await TauBenchAgentLoop._handle_interacting_state(self, data)
                if state == AgentState.TERMINATED:
                    termination = self.eval_termination
            else:
                raise RuntimeError(f"Invalid branch state: {state}")
        raw_score = await data.interaction.calculate_score(data.request_id)
        return self._outcome_score(raw_score), termination

    async def _run_local_branches(
        self, sampling_params: dict[str, Any]
    ) -> dict[str, Any]:
        captured = self._branch_snapshot
        root_id = captured["agent_data"].request_id
        prefix = captured["prefix"]
        candidates = []
        auxiliary_ids = []
        started = time.time()

        for candidate_index in range(int(self._branch_config.candidates)):
            candidate_request_id = f"{root_id}-branch-c{candidate_index}"
            if candidate_index == 0:
                candidate_data = clone_agent_data(
                    captured["root_candidate_data"], candidate_request_id
                )
                candidate_state = captured["root_candidate_state"]
            else:
                candidate_data = clone_agent_data(
                    captured["agent_data"], candidate_request_id
                )
                auxiliary_ids.append(candidate_request_id)
                await candidate_data.interaction.restore_interaction(
                    candidate_request_id, captured["interaction"]
                )
                try:
                    candidate_state = (
                        await TauBenchAgentLoop._handle_generating_state(
                            self, candidate_data, sampling_params
                        )
                    )
                finally:
                    await candidate_data.interaction.finalize_interaction(
                        candidate_request_id
                    )
            if candidate_state == AgentState.TERMINATED:
                record = self._skip_record("candidate_action_not_executable")
                record["auxiliary_trajectory_ids"] = auxiliary_ids
                return record
            candidate_token_ids = list(candidate_data.response_ids)
            if candidate_index == 0 and candidate_token_ids != captured[
                "root_action_token_ids"
            ]:
                raise ValueError(
                    "Candidate 0 tokens differ from the root sampled action"
                )
            candidate_text = self.tokenizer.decode(
                candidate_token_ids, skip_special_tokens=True
            )
            candidate_policy_tokens = len(candidate_token_ids)
            post_candidate = clone_agent_data(candidate_data, candidate_request_id)
            continuations = []

            for repeat_index in range(int(self._branch_config.continuations)):
                branch_id = f"{root_id}-branch-c{candidate_index}-m{repeat_index}"
                auxiliary_ids.append(branch_id)
                await candidate_data.interaction.restore_interaction(
                    branch_id, captured["interaction"]
                )
                branch_data = clone_agent_data(post_candidate, branch_id)
                self.audit_request_id = branch_id
                if branch_data.tool_calls:
                    record_assistant_content(candidate_text)
                repeat_started = time.time()
                try:
                    outcome_reward, termination = await self._continue_branch(
                        branch_data, candidate_state, sampling_params
                    )
                finally:
                    await branch_data.interaction.finalize_interaction(branch_id)

                continuations.append(
                    {
                        "repeat_index": repeat_index,
                        "trajectory_id": branch_id,
                        "outcome_reward": outcome_reward,
                        "termination": termination,
                        "assistant_turns": branch_data.assistant_turns,
                        "user_turns": branch_data.user_turns,
                        "tool_calls": branch_data.total_tool_calls,
                        "suffix_policy_tokens": sum(branch_data.response_mask)
                        - prefix["policy_tokens"]
                        - candidate_policy_tokens,
                        "suffix_total_tokens": len(branch_data.response_mask)
                        - prefix["response_tokens"]
                        - candidate_policy_tokens,
                        "wall_time_s": time.time() - repeat_started,
                    }
                )

            mean_reward = sum(item["outcome_reward"] for item in continuations) / len(
                continuations
            )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_request_id": candidate_request_id,
                    "is_root_action": candidate_index == 0,
                    "candidate_token_ids": candidate_token_ids,
                    "candidate_text": candidate_text,
                    "candidate_policy_tokens": candidate_policy_tokens,
                    "next_state": candidate_state.value,
                    "mean_reward": mean_reward,
                    "continuations": continuations,
                }
            )

        advantages = leave_one_out_advantages(
            [item["mean_reward"] for item in candidates]
        )
        for candidate, advantage in zip(candidates, advantages, strict=True):
            candidate["local_advantage"] = advantage
        root_advantage = candidates[0]["mean_reward"] - sum(
            item["mean_reward"] for item in candidates[1:]
        ) / (len(candidates) - 1)

        sibling_tokens = sum(
            item["candidate_policy_tokens"] for item in candidates[1:]
        )
        suffix_tokens = sum(
            repeat["suffix_policy_tokens"]
            for candidate in candidates
            for repeat in candidate["continuations"]
        )
        branch_user_turns = sum(
            repeat["user_turns"] - prefix["user_turns"]
            for candidate in candidates
            for repeat in candidate["continuations"]
        )
        branch_tool_calls = sum(
            repeat["tool_calls"] - prefix["tool_calls"]
            for candidate in candidates
            for repeat in candidate["continuations"]
        )
        return {
            "schema_version": "local_branch_root_action_v1",
            "record_type": "group",
            "root_trajectory_id": root_id,
            "task_id": self._branch_task_id,
            "split": self._branch_extra_info.get("split"),
            "step": self._branch_trajectory.get("step"),
            "sample_index": self._branch_trajectory.get("sample_index"),
            "rollout_n": self._branch_trajectory.get("rollout_n"),
            "trigger": captured["trigger"],
            "candidates_per_prefix": int(self._branch_config.candidates),
            "continuations_per_candidate": int(self._branch_config.continuations),
            "prefix": prefix,
            "skipped": False,
            "skip_reason": None,
            "root_action_span": captured["root_action_span"],
            "root_action_token_ids": captured["root_action_token_ids"],
            "root_local_advantage": root_advantage,
            "effective": not math.isclose(root_advantage, 0.0, abs_tol=1e-12),
            "candidates": candidates,
            "cost": {
                "root_action_tokens": len(captured["root_action_token_ids"]),
                "sampled_sibling_actions": len(candidates) - 1,
                "sibling_candidate_policy_tokens": sibling_tokens,
                "suffix_policy_tokens": suffix_tokens,
                "branch_user_turns": branch_user_turns,
                "branch_tool_calls": branch_tool_calls,
                "branch_episodes": len(candidates)
                * int(self._branch_config.continuations),
                "wall_time_s": time.time() - started,
            },
            "auxiliary_trajectory_ids": auxiliary_ids,
        }

    def _initialize_training_fields(self, output) -> None:
        if not self._branch_setting("emit_training_fields", False):
            return
        output.extra_fields.update(
            {
                "local_branch_mask": [0] * len(output.response_mask),
                "local_branch_advantages": [0.0] * len(
                    output.response_mask
                ),
                "local_branch_candidate0_token_ids": [],
                "local_branch_record_id": None,
            }
        )

    def _attach_training_fields(self, output, record: dict[str, Any]) -> None:
        if record["skipped"] or not self._branch_setting(
            "emit_training_fields", False
        ):
            return
        start, end = record["root_action_span"]
        if end > len(output.response_mask):
            raise ValueError("Local root action span exceeds the root response")
        if output.response_ids[start:end] != record["root_action_token_ids"]:
            raise ValueError("Local candidate 0 tokens differ from the root response")
        local_mask = [0] * len(output.response_mask)
        local_mask[start:end] = [1] * (end - start)
        if any(
            mask and not assistant
            for mask, assistant in zip(
                local_mask, output.response_mask, strict=True
            )
        ):
            raise ValueError("Local root action mask includes a non-assistant token")
        advantage = float(record["root_local_advantage"])
        output.extra_fields.update(
            {
                "local_branch_mask": local_mask,
                "local_branch_advantages": [
                    advantage if mask else 0.0 for mask in local_mask
                ],
                "local_branch_candidate0_token_ids": record[
                    "root_action_token_ids"
                ],
                "local_branch_record_id": record["root_trajectory_id"],
            }
        )

    async def run(self, sampling_params, **kwargs):
        self._branch_config = self._local_branch_config()
        self._branch_trajectory = kwargs.get("trajectory") or {}
        self._branch_extra_info = kwargs.get("extra_info") or {}
        self._branch_task_id = int(self._branch_extra_info["task_id"])
        self._branch_snapshot = None
        self._branch_attempted = False
        self._branch_trigger = None
        self._branch_skip_reason = None
        self._branch_enabled = bool(self._branch_config.enable)

        trigger_mode = self._branch_setting("trigger", "anchor_turn")
        if trigger_mode not in {"anchor_turn", "critical_prefix"}:
            raise ValueError(
                "local_branch.trigger must be anchor_turn or critical_prefix"
            )
        anchors = [int(turn) for turn in self._branch_config.anchor_turns]
        if not anchors or any(turn < 1 for turn in anchors):
            raise ValueError(
                "local_branch.anchor_turns must contain positive assistant turns"
            )
        if int(self._branch_config.candidates) < 2:
            raise ValueError("local_branch.candidates must be at least 2")
        if int(self._branch_config.continuations) < 1:
            raise ValueError("local_branch.continuations must be positive")
        if self._branch_enabled:
            if self._branch_extra_info.get("split") != "train":
                raise ValueError("Local branching diagnostics are restricted to the train split")
            is_validation = bool(
                self._branch_trajectory.get("validate", False)
            )
            if (
                self._branch_config.validation_only
                and not self.config.trainer.val_only
            ):
                raise ValueError(
                    "Local branching diagnostics require trainer.val_only=true"
                )
            if self._branch_config.validation_only and not is_validation:
                raise ValueError(
                    "Local branching diagnostics require a validation trajectory"
                )
            if not self._branch_config.validation_only and is_validation:
                self._branch_enabled = False
        task_ids = self._branch_config.get("task_ids")
        if task_ids and self._branch_task_id not in {
            int(task_id) for task_id in task_ids
        }:
            self._branch_enabled = False
            skip_reason = "task_not_selected"
        else:
            skip_reason = None

        rollout_n = int(self._branch_trajectory.get("rollout_n", 0))
        self._branch_anchor_turn = anchors[rollout_n % len(anchors)]
        if (
            trigger_mode == "critical_prefix"
            and rollout_n != 0
        ):
            self._branch_enabled = False
            skip_reason = "non_root_rollout"
        output = await super().run(sampling_params, **kwargs)
        self._initialize_training_fields(output)

        if not bool(self._branch_config.enable):
            return output
        if (
            trigger_mode == "critical_prefix"
            and rollout_n != 0
        ):
            return output
        if not self._branch_enabled:
            record = self._skip_record(skip_reason or "disabled")
        elif self._branch_snapshot is None:
            reason = self._branch_skip_reason
            if reason is None:
                termination = getattr(self, "eval_termination", "unknown")
                reason = (
                    "terminal_before_anchor"
                    if termination in {"tool_terminated", "environment_terminated", "interaction_terminated"}
                    else "anchor_not_reached"
                )
            record = self._skip_record(reason)
        else:
            record = await self._run_local_branches(sampling_params)
        append_branch_group(self._branch_output_dir(self._branch_config), record)
        self._attach_training_fields(output, record)
        return output
