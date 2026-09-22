"""Shared tau-bench input protocol and trajectory instrumentation."""

import json
import math
import os
import time
from numbers import Real
from pathlib import Path

import yaml
from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.opsd_feedback import (
    OPSD_FEEDBACK_VERSION,
    build_opsd_feedback,
    empty_opsd_feedback,
)
from tau_bench.envs.airline.wiki import WIKI
from verl.experimental.agent_loop.tool_agent_loop import AgentState


def append_record(record):
    # A single append on the event-loop thread keeps concurrent records intact.
    path = os.environ.get("TRAJECTORIES_PATH") or os.environ["E00_TRAJECTORIES_PATH"]
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class TauBenchAgentLoop(AuditedToolAgentLoop):
    @staticmethod
    def _finite_float(value, name):
        assert isinstance(value, Real) and not isinstance(value, bool), (
            f"{name} must be a numeric scalar"
        )
        value = float(value)
        assert math.isfinite(value), f"{name} must be finite"
        return value

    async def _handle_pending_state(self, data, sampling_params):
        self.eval_data = data
        assert len(data.messages) == 1 and data.messages[0]["role"] == "system", (
            "Expected one policy message"
        )
        assert WIKI in data.messages[0]["content"], "Airline policy missing"
        initial_user = data.interaction.initial_user_message(data.request_id)
        assert isinstance(initial_user, str) and initial_user.strip(), (
            "Empty initial user message"
        )
        data.messages.append({"role": "user", "content": initial_user})
        source = Path(self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
        expected = {
            item["tool_schema"]["function"]["name"]: item["tool_schema"]["function"][
                "parameters"
            ]
            for item in yaml.safe_load(source.read_text())["tools"]
        }
        actual = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in self.tool_schemas
        }
        assert actual == expected, "Tool schema lost fields before rendering the prompt"
        state = await super()._handle_pending_state(data, sampling_params)
        prompt = self.tokenizer.decode(data.prompt_ids)
        assert '"flight_number"' in prompt and '"required"' in prompt
        assert WIKI in prompt and initial_user in prompt, (
            "Policy or initial user lost during tokenization"
        )
        assert len(data.prompt_ids) <= self.prompt_length, (
            "Initial prompt exceeds configured budget"
        )
        assert data.response_mask == [], (
            "Initial user must not contribute to actor loss"
        )
        return state

    async def _handle_generating_state(
        self, data, sampling_params, ignore_termination=False
    ):
        state = await super()._handle_generating_state(
            data, sampling_params, ignore_termination
        )
        if state == AgentState.TERMINATED:
            if len(data.response_mask) >= self.response_length:
                self.eval_termination = "response_budget"
            elif (
                self.max_assistant_turns
                and data.assistant_turns >= self.max_assistant_turns
            ):
                self.eval_termination = "assistant_turn_limit"
            else:
                self.eval_termination = "user_turn_limit"
        return state

    async def _handle_processing_tools_state(self, data):
        state = await super()._handle_processing_tools_state(data)
        if state == AgentState.TERMINATED:
            self.eval_termination = "tool_response_budget"
        return state

    async def _handle_interacting_state(self, data):
        state = await super()._handle_interacting_state(data)
        if state == AgentState.TERMINATED:
            self.eval_termination = "interaction_terminated"
        return state

    async def run(self, sampling_params, **kwargs):
        self.eval_data = None
        self.eval_termination = "unknown"
        started = time.time()
        try:
            output = await super().run(sampling_params, **kwargs)
            data = self.eval_data
            assert data is not None, "Missing trajectory data"

            interaction_reward_mode = getattr(data.interaction, "reward_mode", None)
            assert interaction_reward_mode in {"binary", "prm_lite"}, (
                f"Unsupported interaction reward mode: {interaction_reward_mode!r}"
            )
            reward_mode = self.config.get("tau_bench_reward_mode", "binary")
            assert reward_mode == interaction_reward_mode, (
                "Trainer tau_bench_reward_mode does not match interaction "
                f"reward_mode: {reward_mode!r} != {interaction_reward_mode!r}"
            )
            outcome_reward = self._finite_float(
                output.extra_fields.get("outcome_reward"), "outcome_reward"
            )
            assert outcome_reward in (0.0, 1.0), "outcome_reward must be binary"
            process_score = self._finite_float(
                output.extra_fields.get("process_score"), "process_score"
            )
            assert -0.5 <= process_score <= 0.5, (
                "process_score must be within [-0.5, 0.5]"
            )
            raw_training_reward = self._finite_float(
                output.reward_score, "training_reward"
            )
            expected_reward = outcome_reward
            if reward_mode == "prm_lite":
                expected_reward += 0.3 * process_score
            assert math.isclose(
                raw_training_reward,
                expected_reward,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ), (
                "training_reward does not match configured tau-bench reward: "
                f"{raw_training_reward} != {expected_reward}"
            )

            trajectory = kwargs.get("trajectory") or {}
            is_validation = bool(trajectory.get("validate", False))
            training_reward = (
                outcome_reward if is_validation else float(output.reward_score)
            )
            output.reward_score = training_reward

            opsd_feedback = None
            distillation = self.config.get("distillation", {})
            if (
                distillation.get("teacher_mode") == "self_feedback"
                and not is_validation
            ):
                feedback_mode = distillation.get("feedback_mode", "F0")
                version = distillation.get("feedback_version", OPSD_FEEDBACK_VERSION)
                if feedback_mode == "F2":
                    state = data.interaction.opsd_feedback_state(data.request_id)
                    opsd_feedback = build_opsd_feedback(
                        state,
                        outcome_reward=outcome_reward,
                        termination=self.eval_termination,
                        version=version,
                    )
                elif feedback_mode == "F0":
                    opsd_feedback = empty_opsd_feedback(version)
                else:
                    raise ValueError(
                        f"Unsupported self-feedback mode: {feedback_mode!r}"
                    )
                output.extra_fields["opsd_feedback"] = opsd_feedback

            info = kwargs["extra_info"]
            record = {
                "protocol": info.get("protocol"),
                "step": trajectory.get("step"),
                "validate": is_validation,
                "trajectory_id": data.request_id,
                "task_id": int(info["task_id"]),
                "split": info["split"],
                "score": outcome_reward,
                "training_reward": training_reward,
                "process_score": process_score,
                "reward_mode": reward_mode,
                "termination": self.eval_termination,
                "assistant_turns": data.assistant_turns,
                "user_turns": data.user_turns,
                "assistant_tokens": sum(output.response_mask),
                "tool_calls": data.total_tool_calls,
                "started_at": started,
                "finished_at": time.time(),
                "messages": data.messages,
                # Includes terminal generations that the original loop does not execute.
                "rendered_transcript": self.tokenizer.decode(
                    output.prompt_ids + output.response_ids
                ),
                "schema_verified": True,
            }
            if opsd_feedback is not None:
                record["opsd_feedback"] = opsd_feedback
            append_record(record)
            return output
        finally:
            if self.eval_data is not None and self.eval_data.interaction is not None:
                await self.eval_data.interaction.finalize_interaction(
                    self.eval_data.request_id
                )
