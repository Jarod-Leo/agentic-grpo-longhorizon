"""Regression tests for tau-bench terminal-state handling."""

from types import SimpleNamespace

import pytest

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    make_initial_state,
)
from src.envs.tau_bench_interaction import TauBenchInteraction
from src.envs.tau_bench_tools import TauBenchToolBase
from verl.experimental.agent_loop.tool_agent_loop import AgentState


class FakeEnv:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def step(self, action):
        self.calls.append(action.name)
        return next(self.responses)


def make_tool(name="transfer_to_human_agents"):
    tool = object.__new__(TauBenchToolBase)
    tool.name = name
    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_reward", [0.0, 1.0])
async def test_first_terminal_outcome_is_retained(terminal_reward):
    env = FakeEnv(
        [SimpleNamespace(observation="terminal", reward=terminal_reward, done=True)]
    )
    state = make_initial_state(7)
    interaction = TauBenchInteraction({"reward_mode": "binary"})
    interaction._instance_dict["trajectory"] = {
        "env": env,
        "state": state,
        "initial_user": "request",
    }
    env_token = CURRENT_TAU_ENV.set(env)
    state_token = CURRENT_TAU_STATE.set(state)
    content_token = CURRENT_ASSISTANT_CONTENT.set("I will transfer this request.")
    try:
        _, _, first_metadata = await make_tool().execute("first", {})
        _, _, skipped_metadata = await make_tool("get_user_details").execute(
            "post-terminal", {"user_id": "alice_test_1"}
        )
        should_terminate, _, turn_reward, metadata = await interaction.generate_response(
            "trajectory", [{"role": "assistant", "content": "continue"}]
        )
        final_score = await interaction.calculate_score("trajectory")
    finally:
        CURRENT_ASSISTANT_CONTENT.reset(content_token)
        CURRENT_TAU_STATE.reset(state_token)
        CURRENT_TAU_ENV.reset(env_token)

    assert first_metadata["done"] is True
    assert skipped_metadata["skipped"] == "environment_done"
    assert env.calls == ["transfer_to_human_agents"]
    assert state["num_tool_calls"] == 1
    assert state["total_reward"] == terminal_reward
    assert len(state["action_history"]) == 1
    assert should_terminate is True
    assert turn_reward == terminal_reward
    assert metadata["reason"] == "already_done"
    assert final_score["outcome_score"] == terminal_reward


@pytest.mark.asyncio
async def test_agent_loop_stops_at_terminal_tool(monkeypatch):
    state = make_initial_state(3)
    parent_calls = []

    class Interaction:
        def opsd_feedback_state(self, request_id):
            assert request_id == "trajectory"
            return state

    async def process_tool(self, data):
        parent_calls.append("tool")
        state["done"] = True
        return AgentState.GENERATING

    async def unexpected_call(self, *args, **kwargs):
        parent_calls.append("unexpected")
        return AgentState.GENERATING

    monkeypatch.setattr(
        AuditedToolAgentLoop, "_handle_processing_tools_state", process_tool
    )
    monkeypatch.setattr(
        AuditedToolAgentLoop, "_handle_generating_state", unexpected_call
    )
    monkeypatch.setattr(
        AuditedToolAgentLoop, "_handle_interacting_state", unexpected_call
    )
    loop = object.__new__(TauBenchAgentLoop)
    loop.eval_termination = "unknown"
    data = SimpleNamespace(request_id="trajectory", interaction=Interaction())

    assert await loop._handle_processing_tools_state(data) == AgentState.TERMINATED
    assert loop.eval_termination == "tool_terminated"
    assert await loop._handle_generating_state(data, {}) == AgentState.TERMINATED
    assert loop.eval_termination == "environment_terminated"
    assert await loop._handle_interacting_state(data) == AgentState.TERMINATED
    assert parent_calls == ["tool"]


@pytest.mark.asyncio
async def test_nonterminal_tools_continue():
    env = FakeEnv(
        [
            SimpleNamespace(observation="first", reward=0.0, done=False),
            SimpleNamespace(observation="second", reward=0.0, done=False),
        ]
    )
    state = make_initial_state(9)
    env_token = CURRENT_TAU_ENV.set(env)
    state_token = CURRENT_TAU_STATE.set(state)
    content_token = CURRENT_ASSISTANT_CONTENT.set("Inspect the reservation first.")
    try:
        await make_tool("get_user_details").execute(
            "first", {"user_id": "alice_test_1"}
        )
        await make_tool("get_reservation_details").execute(
            "second", {"reservation_id": "ABC123"}
        )
    finally:
        CURRENT_ASSISTANT_CONTENT.reset(content_token)
        CURRENT_TAU_STATE.reset(state_token)
        CURRENT_TAU_ENV.reset(env_token)

    assert env.calls == ["get_user_details", "get_reservation_details"]
    assert state["done"] is False
    assert state["num_tool_calls"] == 2
    assert len(state["action_history"]) == 2
