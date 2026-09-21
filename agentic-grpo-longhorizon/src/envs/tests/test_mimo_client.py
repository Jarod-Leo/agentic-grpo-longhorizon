"""Exercise missing accounting, malformed replies and bounded simulator retries."""

import json
from types import SimpleNamespace

import pytest
from src.envs.mimo_client import MimoClient, UserSimulatorAPIError
from src.envs.mimo_user_simulator import MimoUserSimulationEnv
from src.evaluation.run_metrics import summarize_api


def reply(content="Please cancel it.", finish="stop", usage=True):
    from openai.types.chat import ChatCompletion

    return ChatCompletion(
        id="test-response",
        created=1,
        model="mimo-v2.5",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        if usage
        else None,
    )


def client_for(tmp_path, monkeypatch, responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    path = tmp_path / "api.jsonl"
    monkeypatch.setenv("MIMO_METRICS_PATH", str(path))
    client = MimoClient(
        {"rpm": 1_000_000},
        sdk=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(client, "_retry_after", lambda exc, attempt: 0)
    return client, calls, path


def test_missing_usage_accepts_valid_text_without_resampling(tmp_path, monkeypatch):
    client, calls, path = client_for(tmp_path, monkeypatch, [reply(usage=False)])
    user = MimoUserSimulationEnv(client, "trajectory")
    assert (
        user.generate_next_message([{"role": "user", "content": "Cancel?"}])
        == "Please cancel it."
    )
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
    messages = [{"role": "user", "content": "Please help."}]
    assert client.complete(messages, "trajectory")[0] == "Please cancel it."
    assert len(calls) == 2 and calls[0]["messages"] is calls[1]["messages"] is messages
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


def test_content_filter_not_retried_or_treated_as_missing_usage(tmp_path, monkeypatch):
    client, calls, path = client_for(
        tmp_path,
        monkeypatch,
        [reply(content=None, finish="content_filter", usage=False)],
    )
    with pytest.raises(UserSimulatorAPIError):
        client.complete([], "trajectory")
    row = json.loads(path.read_text())
    assert (
        len(calls) == 1
        and not row["retry"]
        and row["finish_reason"] == "content_filter"
    )


@pytest.mark.parametrize("status,expected_calls", [(429, 2), (503, 2), (401, 1)])
def test_http_retries_and_sanitized_diagnostics(
    tmp_path, monkeypatch, status, expected_calls
):
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
