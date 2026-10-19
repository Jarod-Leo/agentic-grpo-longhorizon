"""CPU-only tests for tau-bench interaction snapshots."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
)
from src.envs.tau_bench_interaction import TauBenchInteraction


class FakeClient:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class FakeUser:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.trajectory_id = "root"
        self.messages = [
            {"role": "system", "content": "task"},
            {"role": "assistant", "content": {"text": "hello"}},
        ]
        self.total_cost = 0.25


class FakeEnv:
    def __init__(self, client: FakeClient) -> None:
        self.data = {"users": {"alice": {"reservations": ["ABC123"]}}}
        self.actions = [SimpleNamespace(name="get_user_details", kwargs={"user_id": "alice"})]
        self.task = SimpleNamespace(instruction="change flight", outputs=["done"])
        self.user = FakeUser(client)
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, action):
        self.step_calls += 1
        raise AssertionError("restore must not replay actions")

    def reset(self, task_index):
        self.reset_calls += 1
        raise AssertionError("restore must not reset the environment")


@pytest.mark.asyncio
async def test_snapshot_restores_independent_branch_state_without_replay():
    interaction = TauBenchInteraction({"reward_mode": "binary"})
    client = FakeClient()
    root_env = FakeEnv(client)
    root_state = {
        "task_id": 7,
        "total_reward": 1.0,
        "num_tool_calls": 1,
        "num_user_turns": 2,
        "done": False,
        "contaminated": False,
        "action_history": [{"tool": "get_user_details", "parameters": {"user_id": "alice"}}],
        "transferred_to_human": False,
    }
    interaction._instance_dict["root"] = {
        "env": root_env,
        "state": root_state,
        "initial_user": "I need to change my flight",
    }

    snapshot = await interaction.snapshot_interaction("root")

    # Later mutations of the backbone must not alter the saved branch point.
    root_env.data["users"]["alice"]["reservations"].append("LATE99")
    root_env.actions[0].kwargs["user_id"] = "mutated"
    root_env.task.outputs.append("late")
    root_env.user.messages[1]["content"]["text"] = "mutated"
    root_state["action_history"][0]["parameters"]["user_id"] = "mutated"

    CURRENT_ASSISTANT_CONTENT.set("stale assistant turn")
    await interaction.restore_interaction("branch-a", snapshot)
    branch_a = interaction._instance_dict["branch-a"]
    await interaction.restore_interaction("branch-b", snapshot)
    branch_b = interaction._instance_dict["branch-b"]

    assert branch_a["initial_user"] == "I need to change my flight"
    assert branch_a["env"].data["users"]["alice"]["reservations"] == ["ABC123"]
    assert branch_a["env"].actions[0].kwargs == {"user_id": "alice"}
    assert branch_a["env"].task.outputs == ["done"]
    assert branch_a["env"].user.messages[1]["content"] == {"text": "hello"}
    assert branch_a["state"]["action_history"][0]["parameters"] == {"user_id": "alice"}

    # Each branch owns mutable trajectory state but shares the API client.
    branch_a["env"].data["users"]["alice"]["reservations"].append("BRANCHA")
    branch_a["env"].actions[0].kwargs["user_id"] = "branch-a"
    branch_a["env"].task.outputs.append("branch-a")
    branch_a["env"].user.messages.append({"role": "user", "content": "branch-a"})
    branch_a["state"]["action_history"].append({"tool": "branch-a"})

    assert branch_b["env"].data["users"]["alice"]["reservations"] == ["ABC123"]
    assert branch_b["env"].actions[0].kwargs == {"user_id": "alice"}
    assert branch_b["env"].task.outputs == ["done"]
    assert len(branch_b["env"].user.messages) == 2
    assert len(branch_b["state"]["action_history"]) == 1
    assert branch_a["env"].user.client is client
    assert branch_b["env"].user.client is client
    assert branch_a["env"].user.trajectory_id == "branch-a"
    assert branch_b["env"].user.trajectory_id == "branch-b"
    assert root_env.step_calls == root_env.reset_calls == 0

    # The latest restore is bound to this coroutine and scoring sees copied state.
    assert CURRENT_TAU_ENV.get() is branch_b["env"]
    assert CURRENT_TAU_STATE.get() is branch_b["state"]
    assert CURRENT_ASSISTANT_CONTENT.get() is None
    score = await interaction.calculate_score("branch-b")
    assert score["score"] == score["outcome_score"] == 1.0


@pytest.mark.asyncio
async def test_snapshot_rejects_missing_source_and_duplicate_restore_id():
    interaction = TauBenchInteraction({"reward_mode": "binary"})
    with pytest.raises(RuntimeError, match="Missing live tau-bench interaction"):
        await interaction.snapshot_interaction("missing")

    env = FakeEnv(FakeClient())
    interaction._instance_dict["root"] = {
        "env": env,
        "state": {"task_id": 1, "total_reward": 0.0, "action_history": []},
        "initial_user": "hello",
    }
    snapshot = await interaction.snapshot_interaction("root")
    with pytest.raises(RuntimeError, match="already exists"):
        await interaction.restore_interaction("root", snapshot)
