"""Verify that joint PRM-Lite rewards train the actor without changing tau metrics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop


class FakeInteraction:
    def __init__(self, reward_mode="prm_lite"):
        self.reward_mode = reward_mode
        self.finalized = []

    async def finalize_interaction(self, request_id):
        self.finalized.append(request_id)


class FakeTokenizer:
    def decode(self, token_ids):
        return "|".join(str(token_id) for token_id in token_ids)


def make_loop(reward_mode="prm_lite"):
    loop = object.__new__(TauBenchAgentLoop)
    loop.config = {"tau_bench_reward_mode": reward_mode}
    loop.tokenizer = FakeTokenizer()
    return loop


def make_data(interaction):
    return SimpleNamespace(
        request_id="joint-trajectory",
        assistant_turns=3,
        user_turns=2,
        total_tool_calls=4,
        messages=[{"role": "assistant", "content": "terminal answer"}],
        interaction=interaction,
    )


def make_output(training_reward, outcome_reward, process_score):
    return SimpleNamespace(
        reward_score=training_reward,
        prompt_ids=[10, 11],
        response_ids=[20, 21],
        response_mask=[1, 0],
        extra_fields={
            "outcome_reward": outcome_reward,
            "process_score": process_score,
        },
    )


async def run_loop(
    monkeypatch, tmp_path, output, *, validate=False, reward_mode="prm_lite"
):
    interaction = FakeInteraction(reward_mode)
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        self.eval_termination = "interaction_terminated"
        return output

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    trajectory_path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("TRAJECTORIES_PATH", str(trajectory_path))
    result = await make_loop(reward_mode).run(
        {},
        trajectory={"step": 12, "validate": validate},
        extra_info={"protocol": "joint", "task_id": 7, "split": "train"},
    )
    record = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return result, record, interaction


@pytest.mark.asyncio
async def test_prm_lite_training_keeps_shaped_reward_and_binary_metric(
    monkeypatch, tmp_path
):
    output = make_output(0.94, 1.0, -0.2)
    result, record, interaction = await run_loop(monkeypatch, tmp_path, output)

    assert result.reward_score == pytest.approx(0.94)
    assert record["score"] == 1.0
    assert record["training_reward"] == pytest.approx(0.94)
    assert record["process_score"] == pytest.approx(-0.2)
    assert record["reward_mode"] == "prm_lite"
    assert interaction.finalized == ["joint-trajectory"]


@pytest.mark.asyncio
async def test_validation_uses_binary_reward_even_with_process_diagnostic(
    monkeypatch, tmp_path
):
    output = make_output(0.94, 1.0, -0.2)
    result, record, interaction = await run_loop(
        monkeypatch, tmp_path, output, validate=True
    )

    assert result.reward_score == 1.0
    assert record["score"] == 1.0
    assert record["training_reward"] == 1.0
    assert record["process_score"] == pytest.approx(-0.2)
    assert interaction.finalized == ["joint-trajectory"]


@pytest.mark.asyncio
async def test_zero_outcome_retains_process_training_signal(monkeypatch, tmp_path):
    output = make_output(0.15, 0.0, 0.5)
    result, record, _ = await run_loop(monkeypatch, tmp_path, output)

    assert result.reward_score == pytest.approx(0.15)
    assert record["score"] == 0.0
    assert record["training_reward"] == pytest.approx(0.15)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("training_reward", "outcome_reward", "process_score", "message"),
    [
        (0.0, None, 0.0, "outcome_reward must be a numeric scalar"),
        (0.0, 0.5, 0.0, "outcome_reward must be binary"),
        (0.0, 0.0, 0.6, "process_score must be within"),
        (float("nan"), 0.0, 0.0, "training_reward must be finite"),
        (0.1, 0.0, 0.0, "training_reward does not match"),
    ],
)
async def test_malformed_reward_fails_and_still_finalizes(
    monkeypatch,
    tmp_path,
    training_reward,
    outcome_reward,
    process_score,
    message,
):
    output = make_output(training_reward, outcome_reward, process_score)
    interaction = FakeInteraction()
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        return output

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    trajectory_path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("TRAJECTORIES_PATH", str(trajectory_path))

    with pytest.raises(AssertionError, match=message):
        await make_loop().run(
            {},
            trajectory={"validate": False},
            extra_info={"task_id": 7, "split": "train"},
        )

    assert not trajectory_path.exists()
    assert interaction.finalized == ["joint-trajectory"]


@pytest.mark.asyncio
async def test_trainer_and_interaction_reward_modes_must_match(monkeypatch, tmp_path):
    output = make_output(1.0, 1.0, 0.0)
    interaction = FakeInteraction("prm_lite")
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        return output

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    monkeypatch.setenv("TRAJECTORIES_PATH", str(tmp_path / "trajectories.jsonl"))

    with pytest.raises(AssertionError, match="does not match interaction"):
        await make_loop("binary").run(
            {},
            trajectory={"validate": False},
            extra_info={"task_id": 7, "split": "train"},
        )
    assert interaction.finalized == ["joint-trajectory"]
