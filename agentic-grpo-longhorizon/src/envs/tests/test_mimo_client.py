"""Exercise missing accounting, malformed replies and bounded simulator retries."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from src.envs.mimo_client import MimoClient, TeacherOutputTruncatedError, UserSimulatorAPIError
from src.envs.mimo_user_simulator import MimoUserSimulationEnv
from src.evaluation.run_metrics import summarize_api


def reply(content="Please cancel it.", finish="stop", usage=True):
    from openai.types.chat import ChatCompletion

    return ChatCompletion(
        id="test-response",
        created=1,
        model="mimo-v2.6-flash",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14} if usage else None,
    )


def tool_reply(*calls):
    from openai.types.chat import ChatCompletion

    return ChatCompletion(
        id="test-tool-response",
        created=1,
        model="mimo-v2.6-flash",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )


def truncated_tool_reply(name="lookup_order"):
    from openai.types.chat import ChatCompletion

    return ChatCompletion(
        id="test-truncated-tool-response",
        created=1,
        model="mimo-v2.6-flash",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "partial text that must be discarded",
                    "tool_calls": [
                        {
                            "id": "call_partial",
                            "type": "function",
                            "function": {"name": name, "arguments": '{"order_id":'},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 2048, "total_tokens": 2058},
    )


def client_for(tmp_path, monkeypatch, responses, config=None):
    calls = []

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    path = tmp_path / "api.jsonl"
    monkeypatch.setenv("MIMO_METRICS_PATH", str(path))
    client = MimoClient(
        {"rpm": 1_000_000, **(config or {})},
        sdk=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(client, "_retry_after", lambda exc, attempt: 0)
    return client, calls, path


@pytest.mark.parametrize("config", [{}, {"response_format": None}, {"response_format": {"type": "json_object"}}])
def test_response_format_is_optional_and_passed_to_sdk_when_configured(tmp_path, monkeypatch, config):
    client, calls, _path = client_for(tmp_path, monkeypatch, [reply(content='{"answer": "ok"}')], config)
    assert client.complete([{"role": "user", "content": "Return JSON."}], "extraction")[0] == '{"answer": "ok"}'
    assert calls[0]["model"] == "mimo-v2.6-flash"
    if config.get("response_format") is None:
        assert "response_format" not in calls[0]
    else:
        assert calls[0]["response_format"] == {"type": "json_object"}


def test_missing_usage_accepts_valid_text_without_resampling(tmp_path, monkeypatch):
    client, calls, path = client_for(tmp_path, monkeypatch, [reply(usage=False)])
    user = MimoUserSimulationEnv(client, "trajectory")
    assert user.generate_next_message([{"role": "user", "content": "Cancel?"}]) == "Please cancel it."
    assert len(calls) == 1 and not client.fatal.is_set()
    assert len(user.messages) == 1
    assert user.get_total_cost() is None
    row = json.loads(path.read_text())
    assert row["success"] and row["usage"] is None and row["usage_missing"]
    assert row["response_received"] and row["finish_reason"] == "stop"
    # Unknown usage keeps the conservative token reservation; it is not free.
    assert client.limiter.tokens[0]["tokens"] == row["reserved_tokens"]
    summary = summarize_api([row])
    assert summary["requests_missing_usage"] == 1 and not summary["usage_complete"]


@pytest.mark.parametrize("bad", ["empty", "truncated", "missing_choices"])
def test_invalid_response_retries_same_messages_once(tmp_path, monkeypatch, bad):
    malformed = {
        "empty": reply(content=" "),
        "truncated": reply(finish="length"),
        "missing_choices": SimpleNamespace(choices=[], usage=None),
    }[bad]
    client, calls, path = client_for(tmp_path, monkeypatch, [malformed, reply()])
    initial_messages = [{"role": "user", "content": "Please help."}]
    messages = deepcopy(initial_messages)
    assert client.complete(messages, "trajectory")[0] == "Please cancel it."
    assert len(calls) == 2
    assert calls[0]["messages"] == initial_messages
    assert calls[1]["messages"] == initial_messages
    assert messages == initial_messages
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["retry"] and not rows[0]["success"] and rows[0]["error_reason"]
    assert rows[1]["success"] and not client.fatal.is_set()


def test_persistent_invalid_response_aborts_after_four_attempts(tmp_path, monkeypatch):
    client, calls, path = client_for(tmp_path, monkeypatch, [reply(content="")] * 4)
    with pytest.raises(UserSimulatorAPIError, match="empty_or_nontext_content"):
        client.complete([], "trajectory")
    assert len(calls) == 4 and client.fatal.is_set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["retry"] for r in rows] == [True, True, True, False]
    assert not any(r["success"] for r in rows)


def test_configured_attempts_recover_transient_server_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMO_MAX_ATTEMPTS", "8")
    errors = []
    for _ in range(7):
        error = RuntimeError("temporary provider failure")
        error.status_code = 503
        errors.append(error)
    client, calls, path = client_for(tmp_path, monkeypatch, [*errors, reply()])

    assert client.complete([], "trajectory")[0] == "Please cancel it."
    assert len(calls) == 8 and not client.fatal.is_set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["retry"] for row in rows] == [True] * 7 + [False]
    assert rows[-1]["success"]


def test_content_filter_retries_same_messages_and_appends_only_valid_text(tmp_path, monkeypatch):
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [
            reply(content="filtered partial text", finish="content_filter", usage=False),
            reply(content="Valid user reply."),
        ],
    )
    initial_messages = [{"role": "user", "content": "Please help."}]
    user = MimoUserSimulationEnv(client, "trajectory")
    user.messages = deepcopy(initial_messages)

    assert user.generate_next_message(user.messages) == "Valid user reply."
    assert user.messages == [
        *initial_messages,
        {"role": "assistant", "content": "Valid user reply."},
    ]
    assert calls[0]["messages"] == initial_messages
    assert calls[1]["messages"] == initial_messages
    assert not client.fatal.is_set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["retry"] and not rows[0]["success"]
    assert rows[0]["finish_reason"] == rows[0]["error_reason"] == "content_filter"
    assert rows[1]["success"] and rows[1]["finish_reason"] == "stop"
    assert "filtered partial text" not in path.read_text()


def test_teacher_retries_multiple_tool_calls_with_private_single_action_feedback(tmp_path, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Run {name}",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for name in ("lookup_order", "lookup_policy")
    ]
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [
            tool_reply(("lookup_order", {}), ("lookup_policy", {})),
            tool_reply(("lookup_order", {})),
        ],
    )
    initial_messages = [{"role": "user", "content": "Check the order and policy."}]
    messages = deepcopy(initial_messages)

    content, _usage = client.complete(messages, "teacher-trajectory", tools=tools, role="teacher")

    assert json.loads(content.splitlines()[1]) == {"name": "lookup_order", "arguments": {}}
    assert messages == initial_messages
    assert calls[0]["parallel_tool_calls"] is False and calls[0]["tool_choice"] == "auto"
    assert calls[0]["messages"][0]["role"] == "system"
    assert "at most one tool call" in calls[0]["messages"][0]["content"]
    assert calls[0]["messages"][1:] == initial_messages
    assert calls[1]["messages"][:-1] == calls[0]["messages"]
    feedback = calls[1]["messages"][-1]
    assert feedback["role"] == "system"
    assert "multiple tool calls" in feedback["content"]
    assert "None of the proposed tool calls ran" in feedback["content"]
    assert all("tool_calls" not in message and message["role"] != "tool" for message in calls[1]["messages"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["tool_calls"] == 2 and rows[0]["error_reason"] == "multiple_tool_calls"
    assert rows[0]["retry"] and not rows[0]["success"]
    assert rows[1]["tool_calls"] == 1 and rows[1]["success"]


def test_teacher_length_retries_once_at_larger_budget_before_rendering_partial_tool(tmp_path, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [truncated_tool_reply(), tool_reply(("lookup_order", {}))],
        {
            "max_completion_tokens": 2048,
            "teacher_retry_max_tokens": 4096,
            "teacher_rejection_scope": "verified_sft_collection",
        },
    )
    messages = [{"role": "user", "content": "Look up the order."}]

    content, _usage = client.complete(messages, "teacher-length", tools=tools, role="teacher")

    assert json.loads(content.splitlines()[1]) == {"name": "lookup_order", "arguments": {}}
    assert [call["max_completion_tokens"] for call in calls] == [2048, 4096]
    assert calls[1]["messages"] == calls[0]["messages"]
    assert all("partial text" not in str(message) for message in calls[1]["messages"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["max_completion_tokens"] for row in rows] == [2048, 4096]
    assert rows[0]["error_reason"] == "teacher_output_truncated" and rows[0]["retry"]
    assert rows[0]["content_chars"] is None
    assert all(row["teacher_rejection_scope"] == "verified_sft_collection" for row in rows)


def test_exhausted_teacher_length_is_local_and_does_not_abort_other_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMO_MAX_ATTEMPTS", "1")
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [reply(content="first partial", finish="length"), reply(content="second partial", finish="length"), reply()],
        {
            "max_completion_tokens": 2048,
            "teacher_retry_max_tokens": 4096,
            "teacher_rejection_scope": "verified_sft_collection",
        },
    )

    with pytest.raises(TeacherOutputTruncatedError, match="teacher_output_truncated"):
        client.complete([{"role": "user", "content": "Help."}], "teacher-truncated", role="teacher")

    assert not client.fatal.is_set()
    assert client.complete([{"role": "user", "content": "Continue."}], "next-request")[0] == "Please cancel it."
    assert [call["max_completion_tokens"] for call in calls] == [2048, 4096, 2048]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["retry"] for row in rows[:2]] == [True, False]
    assert [row["error_reason"] for row in rows[:2]] == [
        "teacher_output_truncated",
        "teacher_output_truncated",
    ]
    assert rows[2]["role"] == "user" and rows[2]["success"]


def test_persistent_content_filter_aborts_after_four_attempts(tmp_path, monkeypatch):
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [reply(content="filtered partial text", finish="content_filter", usage=False)] * 4,
    )
    with pytest.raises(UserSimulatorAPIError, match="reason=content_filter"):
        client.complete([], "trajectory")
    assert len(calls) == 4 and client.fatal.is_set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["retry"] for row in rows] == [True, True, True, False]
    assert not any(row["success"] for row in rows)
    assert all(row["error_reason"] == "content_filter" for row in rows)
    assert "filtered partial text" not in path.read_text()


@pytest.mark.parametrize("status,expected_calls", [(429, 2), (503, 2), (401, 1)])
def test_http_retries_and_sanitized_diagnostics(tmp_path, monkeypatch, status, expected_calls):
    error = RuntimeError("private-provider-error api-key=DO_NOT_LOG")
    error.status_code = status
    client, calls, path = client_for(tmp_path, monkeypatch, [error, reply()])
    if status == 401:
        with pytest.raises(UserSimulatorAPIError) as caught:
            client.complete([], "trajectory")
        assert "DO_NOT_LOG" not in str(caught.value)
    else:
        assert client.complete([], "trajectory")[0]
    assert len(calls) == expected_calls
    assert "DO_NOT_LOG" not in path.read_text()
