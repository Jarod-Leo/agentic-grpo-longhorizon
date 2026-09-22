"""CPU lifecycle tests for the E00 evaluation-only agent loop."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.baseline_eval_agent_loop import BaselineEvalAgentLoop


class FakeInteraction:
    def __init__(self):
        self.reward_mode = "binary"
        self.finalized = []

    async def finalize_interaction(self, request_id):
        self.finalized.append(request_id)


class FakeTokenizer:
    def decode(self, token_ids):
        return "|".join(str(token_id) for token_id in token_ids)


def make_loop():
    loop = object.__new__(BaselineEvalAgentLoop)
    loop.config = {}
    loop.tokenizer = FakeTokenizer()
    return loop


def make_data(interaction):
    return SimpleNamespace(
        request_id="trajectory-1",
        assistant_turns=3,
        user_turns=2,
        total_tool_calls=4,
        messages=[{"role": "assistant", "content": "terminal answer"}],
        interaction=interaction,
    )


def make_output(reward_score):
    return SimpleNamespace(
        reward_score=reward_score,
        prompt_ids=[10, 11],
        response_ids=[20, 21, 99],
        response_mask=[1, 0, 1],
        extra_fields={"outcome_reward": reward_score, "process_score": 0.0},
    )


@pytest.mark.asyncio
async def test_zero_score_is_recorded_with_terminal_generation_and_finalized(
    monkeypatch, tmp_path
):
    interaction = FakeInteraction()
    data = make_data(interaction)
    output = make_output(0)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        self.eval_termination = "user_turn_limit"
        return output

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    trajectory_path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("E00_TRAJECTORIES_PATH", str(trajectory_path))

    result = await make_loop().run({}, extra_info={"task_id": 7, "split": "dev"})

    assert result is output
    record = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert record["score"] == 0
    assert record["rendered_transcript"] == "10|11|20|21|99"
    assert record["rendered_transcript"].endswith("99")
    assert interaction.finalized == ["trajectory-1"]


@pytest.mark.asyncio
async def test_missing_reward_fails_instead_of_becoming_zero(monkeypatch, tmp_path):
    interaction = FakeInteraction()
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        return make_output(None)

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    trajectory_path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("E00_TRAJECTORIES_PATH", str(trajectory_path))

    with pytest.raises(AssertionError, match="outcome_reward must be a numeric scalar"):
        await make_loop().run({}, extra_info={"task_id": 7, "split": "dev"})

    assert not trajectory_path.exists()
    assert interaction.finalized == ["trajectory-1"]


@pytest.mark.asyncio
async def test_parent_error_propagates_and_still_finalizes(monkeypatch, tmp_path):
    interaction = FakeInteraction()
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        raise RuntimeError("generation failed")

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    trajectory_path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("E00_TRAJECTORIES_PATH", str(trajectory_path))

    with pytest.raises(RuntimeError, match="generation failed"):
        await make_loop().run({}, extra_info={"task_id": 7, "split": "dev"})

    assert not trajectory_path.exists()
    assert interaction.finalized == ["trajectory-1"]
