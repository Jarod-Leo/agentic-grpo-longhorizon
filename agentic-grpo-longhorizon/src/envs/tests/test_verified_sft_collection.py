"""CPU checks for teacher tool history and Qwen response boundaries."""

import json

import pytest

pytest.importorskip("ray")

from src.envs.mimo_client import MimoResponseError
from src.envs.verified_sft_agent_loop import openai_teacher_messages, render_teacher_response_tokens


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
