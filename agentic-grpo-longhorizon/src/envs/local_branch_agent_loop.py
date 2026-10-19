"""Frozen-policy local branching diagnostics for tau-bench evaluation."""

from __future__ import annotations

import math
import time
from typing import Any

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from src.envs.tau_bench_context import CURRENT_TAU_STATE
from src.envs.tau_bench_interaction import record_assistant_content
from src.evaluation.local_branching import (
    append_branch_group,
    clone_agent_data,
    leave_one_out_advantages,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


class LocalBranchDiagnosticAgentLoop(TauBenchAgentLoop):
    """Run K x M suffix branches after an unchanged validation backbone."""

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
            "schema_version": "local_branch_diagnostic_v1",
            "record_type": "group",
            "root_trajectory_id": getattr(self.eval_data, "request_id", None),
            "task_id": self._branch_task_id,
            "split": self._branch_extra_info.get("split"),
            "step": self._branch_trajectory.get("step"),
            "sample_index": self._branch_trajectory.get("sample_index"),
            "rollout_n": self._branch_trajectory.get("rollout_n"),
            "requested_anchor_turn": self._branch_anchor_turn,
            "captured_anchor_turn": None,
            "candidates_per_prefix": int(self._branch_config.candidates),
            "continuations_per_candidate": int(self._branch_config.continuations),
            "skipped": True,
            "skip_reason": reason,
            "candidates": [],
        }

    def _eligible_anchor(self, data) -> bool:
        state = CURRENT_TAU_STATE.get() or {}
        if state.get("done") or state.get("contaminated"):
            return False
        if self._branch_snapshot is not None:
            return False
        if data.assistant_turns + 1 != self._branch_anchor_turn:
            return False
        if (
            self.max_assistant_turns
            and data.assistant_turns + 1 >= self.max_assistant_turns
        ):
            return False
        if self.max_user_turns and data.user_turns + 1 >= self.max_user_turns:
            return False
        return len(data.response_mask) < self.response_length

    async def _handle_generating_state(
        self, data, sampling_params, ignore_termination=False
    ):
        if self._branch_enabled and self._eligible_anchor(data):
            self._branch_snapshot = {
                "agent_data": clone_agent_data(data, data.request_id),
                "interaction": await data.interaction.snapshot_interaction(
                    data.request_id
                ),
                "prefix": {
                    "assistant_turns": data.assistant_turns,
                    "user_turns": data.user_turns,
                    "response_tokens": len(data.response_mask),
                    "policy_tokens": sum(data.response_mask),
                    "tool_calls": data.total_tool_calls,
                    "message_count": len(data.messages),
                },
            }
        return await super()._handle_generating_state(
            data, sampling_params, ignore_termination
        )

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
                state = await AuditedToolAgentLoop._handle_generating_state(
                    self, data, sampling_params
                )
                if state == AgentState.TERMINATED:
                    if len(data.response_mask) >= self.response_length:
                        termination = "response_budget"
                    elif (
                        self.max_assistant_turns
                        and data.assistant_turns >= self.max_assistant_turns
                    ):
                        termination = "assistant_turn_limit"
                    else:
                        termination = "user_turn_limit"
            elif state == AgentState.PROCESSING_TOOLS:
                state = await ToolAgentLoop._handle_processing_tools_state(self, data)
                if state == AgentState.TERMINATED:
                    termination = "tool_response_budget"
            elif state == AgentState.INTERACTING:
                state = await ToolAgentLoop._handle_interacting_state(self, data)
                if state == AgentState.TERMINATED:
                    termination = "interaction_terminated"
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
        started = time.time()

        for candidate_index in range(int(self._branch_config.candidates)):
            candidate_request_id = f"{root_id}-branch-c{candidate_index}"
            candidate_data = clone_agent_data(
                captured["agent_data"], candidate_request_id
            )
            await candidate_data.interaction.restore_interaction(
                candidate_request_id, captured["interaction"]
            )
            try:
                candidate_state = await AuditedToolAgentLoop._handle_generating_state(
                    self, candidate_data, sampling_params
                )
            finally:
                await candidate_data.interaction.finalize_interaction(
                    candidate_request_id
                )
            candidate_token_ids = list(candidate_data.response_ids)
            candidate_text = self.tokenizer.decode(
                candidate_token_ids, skip_special_tokens=True
            )
            candidate_policy_tokens = len(candidate_token_ids)
            post_candidate = clone_agent_data(candidate_data, candidate_request_id)
            continuations = []

            for repeat_index in range(int(self._branch_config.continuations)):
                branch_id = f"{root_id}-branch-c{candidate_index}-m{repeat_index}"
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

        candidate_tokens = sum(item["candidate_policy_tokens"] for item in candidates)
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
            "schema_version": "local_branch_diagnostic_v1",
            "record_type": "group",
            "root_trajectory_id": root_id,
            "task_id": self._branch_task_id,
            "split": self._branch_extra_info.get("split"),
            "step": self._branch_trajectory.get("step"),
            "sample_index": self._branch_trajectory.get("sample_index"),
            "rollout_n": self._branch_trajectory.get("rollout_n"),
            "requested_anchor_turn": self._branch_anchor_turn,
            "captured_anchor_turn": self._branch_anchor_turn,
            "candidates_per_prefix": int(self._branch_config.candidates),
            "continuations_per_candidate": int(self._branch_config.continuations),
            "prefix": prefix,
            "skipped": False,
            "skip_reason": None,
            "effective": len({item["mean_reward"] for item in candidates}) > 1,
            "candidates": candidates,
            "cost": {
                "candidate_policy_tokens": candidate_tokens,
                "suffix_policy_tokens": suffix_tokens,
                "branch_user_turns": branch_user_turns,
                "branch_tool_calls": branch_tool_calls,
                "branch_episodes": len(candidates)
                * int(self._branch_config.continuations),
                "wall_time_s": time.time() - started,
            },
        }

    async def run(self, sampling_params, **kwargs):
        self._branch_config = self._local_branch_config()
        self._branch_trajectory = kwargs.get("trajectory") or {}
        self._branch_extra_info = kwargs.get("extra_info") or {}
        self._branch_task_id = int(self._branch_extra_info["task_id"])
        self._branch_snapshot = None
        self._branch_enabled = bool(self._branch_config.enable)

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
            if self._branch_config.validation_only and not self.config.trainer.val_only:
                raise ValueError(
                    "Local branching diagnostics require trainer.val_only=true"
                )
            if not self._branch_trajectory.get("validate", False):
                raise ValueError(
                    "Local branching diagnostics require a validation trajectory"
                )
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
        output = await super().run(sampling_params, **kwargs)

        if not bool(self._branch_config.enable):
            return output
        if not self._branch_enabled:
            record = self._skip_record(skip_reason or "disabled")
        elif self._branch_snapshot is None:
            record = self._skip_record("anchor_not_reached_or_insufficient_budget")
        else:
            record = await self._run_local_branches(sampling_params)
        append_branch_group(self._branch_output_dir(self._branch_config), record)
        return output
