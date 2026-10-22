"""CPU checks for teacher tool history and Qwen response boundaries."""

import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from src.envs.mimo_client import MimoResponseError, TeacherOutputTruncatedError
from src.envs.verified_sft_agent_loop import (
    VerifiedSFTTauBenchAgentLoop,
    openai_teacher_messages,
    render_teacher_response_tokens,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


def test_visible_hermes_history_becomes_single_openai_tool_exchange():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "request"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"lookup","arguments":{"id":"A"}}</tool_call>',
        },
        {"role": "tool", "content": "result"},
    ]
    converted = openai_teacher_messages(messages)
    call = converted[2]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"id": "A"}
    assert converted[3]["tool_call_id"] == call["id"]
    assert all(set(message) <= {"role", "content", "tool_calls", "tool_call_id"} for message in converted)


def test_multiple_historical_tool_calls_are_rejected():
    call = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    with pytest.raises(MimoResponseError, match="history_multiple_tool_calls"):
        openai_teacher_messages([{"role": "assistant", "content": call + call}])


class BoundaryTokenizer:
    eos_token_id = 151645

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        if add_generation_prompt:
            return [10, 11, 12]
        assert messages[-1] == {
            "role": "assistant",
            "content": "<think>\n\n</think>\n\nanswer",
        }
        return [10, 11, 12, 20, 151645, 198]


def test_teacher_response_uses_template_suffix_including_assistant_end_token():
    tokens = render_teacher_response_tokens(
        BoundaryTokenizer(),
        [{"role": "user", "content": "request"}],
        [],
        "answer",
        {"enable_thinking": False},
    )
    assert tokens == [20, 151645]


@pytest.mark.asyncio
async def test_collection_rejects_exhausted_teacher_length_on_this_trajectory_only():
    class TruncatedTeacher:
        def __init__(self):
            self.calls = []

        def complete(self, *args):
            self.calls.append(args)
            raise TeacherOutputTruncatedError()

    teacher = TruncatedTeacher()
    loop = object.__new__(VerifiedSFTTauBenchAgentLoop)
    loop.teacher_takeover = True
    loop.collection_failure = None
    loop.loop = asyncio.get_running_loop()
    loop.teacher_client = teacher
    loop.tool_schemas = []
    loop._environment_done = lambda _data: False
    audits = []
    loop._write_audit = audits.append
    data = SimpleNamespace(
        messages=[{"role": "user", "content": "request"}],
        request_id="teacher-truncated",
        response_mask=[],
        assistant_turns=0,
    )

    state = await loop._handle_generating_state(data, {})

    assert state == AgentState.TERMINATED
    assert loop.collection_failure == "teacher_output_truncated"
    assert loop.eval_termination == "teacher_output_truncated"
    assert teacher.calls[0][1] == "teacher-truncated"
    assert teacher.calls[0][-1] == "teacher"
    assert audits == [
        {
            "event": "teacher_rejected",
            "turn": 1,
            "reason": "teacher_output_truncated",
        }
    ]


@pytest.mark.asyncio
async def test_parent_loop_preserves_prompt_when_terminated_without_response_tokens():
    class EmptyResponseLoop(ToolAgentLoop):
        async def _handle_pending_state(self, data, _sampling_params):
            data.prompt_ids = [10, 11, 12]
            return AgentState.TERMINATED

    loop = object.__new__(EmptyResponseLoop)
    loop.interaction_config_file = None
    loop.response_length = 32

    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "request"}])

    assert output.prompt_ids == [10, 11, 12]
    assert output.response_ids == []
    assert output.response_mask == []


def test_collection_subset_updates_summary_task_expectations(tmp_path, monkeypatch):
    metadata = {"train_task_ids": list(range(40)), "test_task_ids": [48, 49]}
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    script = Path(__file__).resolve().parents[3] / "scripts/train/sft/prepare_verified_collection.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(tmp_path), "--mode", "teacher",
                                     "--task-ids", "0,1,4,6,13,45", "--samples", "2"])
    runpy.run_path(str(script), run_name="__main__")
    actual = json.loads((tmp_path / "run.json").read_text())
    assert actual["train_task_ids"] == [0, 1, 4, 6, 13, 45]
    assert actual["test_task_ids"] == []
    assert actual["expected_trajectories"] == 12
