"""Verified teacher and student-correction collection on the shared tau-bench loop."""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

from src.envs.mimo_client import MimoClient, MimoResponseError
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.workers.rollout.replica import TokenOutput


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_NONEMPTY_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_LEADING_EMPTY_THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)
_QWEN3_NON_THINKING_PREFILL = "<think>\n\n</think>\n\n"


class _FixedTokenServer:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    async def generate(self, **_kwargs) -> TokenOutput:
        return TokenOutput(token_ids=self.token_ids)


def openai_teacher_messages(messages: list[dict]) -> list[dict]:
    """Convert the visible Hermes transcript to strict OpenAI tool history."""
    converted = []
    pending_tool_call = None
    for index, source in enumerate(messages):
        message = deepcopy(source)
        message.pop("loss_mask", None)
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            blocks = _TOOL_CALL_RE.findall(content or "")
            if len(blocks) > 1:
                raise MimoResponseError("history_multiple_tool_calls", retryable=False)
            if blocks:
                try:
                    call = json.loads(blocks[0])
                    name = call["name"]
                    arguments = call["arguments"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise MimoResponseError("history_malformed_tool_call", retryable=False) from exc
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise MimoResponseError("history_malformed_tool_call", retryable=False)
                pending_tool_call = f"call_{index}"
                remaining = _TOOL_CALL_RE.sub("", content).strip()
                converted.append(
                    {
                        "role": "assistant",
                        "content": remaining or None,
                        "tool_calls": [
                            {
                                "id": pending_tool_call,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
            else:
                pending_tool_call = None
                converted.append({"role": "assistant", "content": content})
        elif role == "tool":
            if pending_tool_call is None:
                raise MimoResponseError("history_orphan_tool_result", retryable=False)
            converted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": pending_tool_call,
                }
            )
            pending_tool_call = None
        else:
            pending_tool_call = None
            converted.append({"role": role, "content": content})
    if pending_tool_call is not None:
        raise MimoResponseError("history_missing_tool_result", retryable=False)
    return converted


def render_teacher_response_tokens(tokenizer, messages, tools, content, template_kwargs):
    """Render the response suffix, including Qwen's assistant end-of-turn token."""
    prefix = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    template_content = content
    if template_kwargs.get("enable_thinking") is False:
        template_content = _QWEN3_NON_THINKING_PREFILL + content
    completed = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": template_content}],
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if completed[: len(prefix)] != prefix:
        raise ValueError("Teacher response does not extend the Qwen generation prompt")
    response = completed[len(prefix) :]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id not in response:
        raise ValueError("Teacher response is missing the assistant end-of-turn token")
    return response[: response.index(eos_token_id) + 1]


class VerifiedSFTTauBenchAgentLoop(TauBenchAgentLoop):
    """Collect teacher-only or post-error correction targets without replaying tools."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        super().init_class(config, tokenizer, processor, **kwargs)
        if not bool(config.trainer.get("val_only", False)):
            raise ValueError("Verified SFT collection requires trainer.val_only=true")
        collection = config.get("verified_sft_collection", {})
        cls.collection_mode = str(collection.get("mode", "teacher"))
        if cls.collection_mode not in {"teacher", "correction"}:
            raise ValueError(f"Unsupported verified SFT collection mode: {cls.collection_mode}")
        cls.teacher_max_tokens = int(collection.get("teacher_max_tokens", 1024))
        teacher_config = {
            "user_model": collection.get("teacher_model", "mimo-v2.6-pro"),
            "user_base_url": collection.get("teacher_base_url", "https://api.xiaomimimo.com/v1"),
            "max_completion_tokens": cls.teacher_max_tokens,
            "rpm": int(collection.get("rpm", 100)),
            "tpm": int(collection.get("tpm", 10_000_000)),
            "max_inflight": int(collection.get("max_inflight", 32)),
            "limiter_group": collection.get("limiter_group", "verified_sft_mimo_v26"),
        }
        cls.teacher_client = MimoClient(teacher_config)

    def _write_collection_failure(self, kwargs, reason: str) -> None:
        path = os.environ.get("VERIFIED_SFT_FAILURES_PATH")
        if not path:
            return
        info = kwargs.get("extra_info", {})
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "mode": self.collection_mode,
                        "task_id": info.get("task_id"),
                        "reason": reason,
                    }
                )
                + "\n"
            )

    async def run(self, sampling_params, **kwargs):
        trajectory = kwargs.get("trajectory") or {}
        if not bool(trajectory.get("validate", False)):
            raise ValueError("Verified SFT collection accepts validation trajectories only")
        self.teacher_takeover = self.collection_mode == "teacher"
        self.teacher_response_offset = 0 if self.teacher_takeover else None
        self.collection_failure = None
        self.teacher_usage = []
        try:
            output = await super().run(sampling_params, **kwargs)
            # Student and fixed teacher generations do not share a log-probability source.
            output.response_logprobs = None
            return output
        except Exception as exc:
            reason = getattr(exc, "reason", type(exc).__name__)
            self._write_collection_failure(kwargs, str(reason))
            raise

    async def _teacher_tokens(self, data) -> list[int] | None:
        messages = openai_teacher_messages(data.messages)
        content, usage = await self.loop.run_in_executor(
            None,
            self.teacher_client.complete,
            messages,
            data.request_id,
            self.tool_schemas,
            "teacher",
        )
        self.teacher_usage.append(usage)
        if any(match.strip() for match in _NONEMPTY_THINK_RE.findall(content)):
            self.collection_failure = "teacher_nonempty_think"
            return None
        content = _LEADING_EMPTY_THINK_RE.sub("", content, count=1)
        token_ids = await self.loop.run_in_executor(
            None,
            lambda: render_teacher_response_tokens(
                self.tokenizer,
                data.messages,
                self.tool_schemas,
                content,
                self.apply_chat_template_kwargs,
            ),
        )
        remaining = self.response_length - len(data.response_mask)
        if len(token_ids) > self.teacher_max_tokens:
            self.collection_failure = "teacher_turn_token_budget"
            return None
        if len(token_ids) > remaining:
            self.collection_failure = "trajectory_token_budget"
            return None
        return token_ids

    async def _handle_generating_state(self, data, sampling_params, ignore_termination=False):
        if not self.teacher_takeover:
            return await super()._handle_generating_state(data, sampling_params, ignore_termination)
        if self._environment_done(data):
            self.eval_termination = "environment_terminated"
            return AgentState.TERMINATED

        token_ids = await self._teacher_tokens(data)
        if token_ids is None:
            self.eval_termination = self.collection_failure
            self._write_audit(
                {
                    "event": "teacher_rejected",
                    "turn": data.assistant_turns + 1,
                    "reason": self.collection_failure,
                }
            )
            return AgentState.TERMINATED

        original_server = self.server_manager
        self.server_manager = _FixedTokenServer(token_ids)
        try:
            return await super()._handle_generating_state(data, sampling_params, ignore_termination)
        finally:
            self.server_manager = original_server

    async def _handle_processing_tools_state(self, data):
        errors_before = data.total_errors
        state = await super()._handle_processing_tools_state(data)
        if (
            self.collection_mode == "correction"
            and not self.teacher_takeover
            and data.total_errors > errors_before
            and state == AgentState.GENERATING
            and not self._environment_done(data)
        ):
            self.teacher_takeover = True
            self.teacher_response_offset = len(data.response_mask)
        return state

    def _trajectory_record_extra(self, data, output):
        state = data.interaction.opsd_feedback_state(data.request_id)
        response_mask = list(output.response_mask)
        if self.teacher_response_offset is None:
            response_mask = [0] * len(response_mask)
        else:
            response_mask[: self.teacher_response_offset] = [0] * min(
                self.teacher_response_offset, len(response_mask)
            )
        input_ids = list(output.prompt_ids) + list(output.response_ids)
        loss_mask = [0] * len(output.prompt_ids) + response_mask
        return {
            "sft_mode": self.collection_mode,
            "teacher_takeover": self.teacher_takeover,
            "loss_start": (
                len(output.prompt_ids) + self.teacher_response_offset
                if self.teacher_response_offset is not None
                else None
            ),
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "contaminated": bool(state.get("contaminated", False)),
            "collection_failure": self.collection_failure,
            "teacher_usage": self.teacher_usage,
        }
