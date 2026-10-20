"""CPU tests for experiment C local branching diagnostics."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("ray")

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.tau_bench_context import CURRENT_TAU_STATE
from src.envs.local_branch_agent_loop import LocalBranchDiagnosticAgentLoop
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from src.evaluation.local_branching import clone_agent_data, leave_one_out_advantages
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState


class FakeInteraction:
    def __init__(self):
        self.restored = []
        self.finalized = []

    async def restore_interaction(self, request_id, snapshot):
        self.restored.append((request_id, snapshot))

    async def finalize_interaction(self, request_id):
        self.finalized.append(request_id)


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def make_agent_data(interaction):
    data = AgentData(
        messages=[{"role": "system", "content": "policy"}],
        image_data=None,
        metrics={"prefix": 1.0},
        request_id="root-trajectory",
        tools_kwargs={},
        interaction=interaction,
    )
    data.prompt_ids = [1, 2]
    data.response_ids = [2]
    data.response_mask = [1]
    data.response_logprobs = [0.0]
    data.assistant_turns = 1
    data.user_turns = 1
    return data


def make_root_candidate(prefix_data, token=99):
    candidate = clone_agent_data(prefix_data, prefix_data.request_id)
    candidate.response_ids = [token]
    candidate.prompt_ids.append(token)
    candidate.response_mask.append(1)
    candidate.assistant_turns += 1
    candidate.messages.append({"role": "assistant", "content": str(token)})
    candidate.tool_calls = []
    return candidate


def test_clone_agent_data_keeps_service_but_isolates_mutable_state():
    interaction = FakeInteraction()
    source = make_agent_data(interaction)

    cloned = clone_agent_data(source, "branch")
    cloned.messages.append({"role": "user", "content": "changed"})
    cloned.prompt_ids.append(3)

    assert cloned.interaction is interaction
    assert cloned.request_id == "branch"
    assert len(source.messages) == 1
    assert source.prompt_ids == [1, 2]
    assert cloned.metrics == {}


def test_leave_one_out_advantages_are_zero_for_equal_candidate_rewards():
    assert leave_one_out_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0] * 4
    assert leave_one_out_advantages([1.0, 0.0, 0.5, 0.5]) == pytest.approx(
        [2 / 3, -2 / 3, 0.0, 0.0]
    )


@pytest.mark.parametrize("state", [{"done": True}, {"contaminated": True}])
def test_terminal_or_contaminated_prefix_is_not_branched(state):
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    token = CURRENT_TAU_STATE.set(state)
    try:
        assert not loop._eligible_anchor(None)
    finally:
        CURRENT_TAU_STATE.reset(token)


def test_anchor_requires_one_more_assistant_and_user_turn_of_budget():
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    loop._branch_snapshot = None
    loop._branch_anchor_turn = 2
    loop.max_assistant_turns = 15
    loop.max_user_turns = 15
    loop.response_length = 100
    data = make_agent_data(FakeInteraction())

    assert loop._eligible_anchor(data)
    data.user_turns = 14
    assert not loop._eligible_anchor(data)
    data.user_turns = 1
    data.assistant_turns = 14
    loop._branch_anchor_turn = 15
    assert not loop._eligible_anchor(data)


@pytest.mark.asyncio
async def test_branch_stops_after_terminal_tool_without_generation_or_user_call(monkeypatch):
    state = {"done": False}
    calls = {"tool": 0, "generation": 0, "interaction": 0}

    class TerminalInteraction:
        def opsd_feedback_state(self, request_id):
            return state

        async def calculate_score(self, request_id):
            return {"outcome_score": 0.0}

    async def fake_tool_handler(self, data):
        calls["tool"] += 1
        state["done"] = True
        return AgentState.GENERATING

    async def unexpected_generation(self, data, sampling_params, ignore_termination=False):
        calls["generation"] += 1
        return AgentState.TERMINATED

    async def unexpected_interaction(self, data):
        calls["interaction"] += 1
        return AgentState.TERMINATED

    monkeypatch.setattr(
        AuditedToolAgentLoop, "_handle_processing_tools_state", fake_tool_handler
    )
    monkeypatch.setattr(
        TauBenchAgentLoop, "_handle_generating_state", unexpected_generation
    )
    monkeypatch.setattr(
        TauBenchAgentLoop, "_handle_interacting_state", unexpected_interaction
    )
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    data = SimpleNamespace(
        interaction=TerminalInteraction(), request_id="terminal-branch"
    )

    reward, termination = await loop._continue_branch(
        data, AgentState.PROCESSING_TOOLS, {}
    )

    assert reward == 0.0
    assert termination == "tool_terminated"
    assert calls == {"tool": 1, "generation": 0, "interaction": 0}


@pytest.mark.asyncio
async def test_branch_group_generates_each_candidate_once_and_runs_k_times_m(
    monkeypatch,
):
    interaction = FakeInteraction()
    prefix_data = make_agent_data(interaction)
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    loop.tokenizer = FakeTokenizer()
    loop._branch_config = SimpleNamespace(candidates=4, continuations=2)
    loop._branch_snapshot = {
        "agent_data": prefix_data,
        "interaction": object(),
        "trigger": "after_tool_error",
        "root_candidate_data": make_root_candidate(prefix_data),
        "root_candidate_state": AgentState.INTERACTING,
        "root_action_token_ids": [99],
        "root_action_span": [1, 2],
        "prefix": {
            "assistant_turns": 1,
            "user_turns": 1,
            "response_tokens": 1,
            "policy_tokens": 1,
            "tool_calls": 0,
            "message_count": 1,
        },
    }
    loop._branch_task_id = 7
    loop._branch_extra_info = {"split": "train"}
    loop._branch_trajectory = {"step": 0, "sample_index": 3, "rollout_n": 1}
    loop._branch_anchor_turn = 2
    generated = []
    continued_candidate_tokens = {}

    async def fake_generate(self, data, sampling_params, ignore_termination=False):
        candidate_index = int(data.request_id.rsplit("c", 1)[1])
        token = 100 + candidate_index
        generated.append(data.request_id)
        data.response_ids = [token]
        data.prompt_ids.append(token)
        data.response_mask.append(1)
        data.assistant_turns += 1
        data.messages.append({"role": "assistant", "content": str(token)})
        data.tool_calls = []
        return AgentState.INTERACTING

    async def fake_continue(self, data, initial_state, sampling_params):
        candidate_index = int(data.request_id.split("-branch-c", 1)[1].split("-", 1)[0])
        continued_candidate_tokens.setdefault(candidate_index, []).append(
            data.response_ids[:]
        )
        data.response_mask.append(1)
        data.assistant_turns += 1
        return float(candidate_index % 2), "interaction_terminated"

    monkeypatch.setattr(TauBenchAgentLoop, "_handle_generating_state", fake_generate)
    loop._continue_branch = MethodType(fake_continue, loop)

    record = await loop._run_local_branches({})

    assert len(generated) == 3
    assert len(interaction.restored) == 3 + 4 * 2
    assert interaction.finalized == [
        request_id for request_id, _ in interaction.restored
    ]
    assert record["cost"]["branch_episodes"] == 8
    assert record["effective"] is True
    assert record["candidates"][0]["candidate_token_ids"] == [99]
    assert record["root_action_token_ids"] == [99]
    assert [item["mean_reward"] for item in record["candidates"]] == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    for candidate_index, repeated_tokens in continued_candidate_tokens.items():
        expected = 99 if candidate_index == 0 else 100 + candidate_index
        assert repeated_tokens == [[expected], [expected]]


@pytest.mark.asyncio
async def test_branch_continuation_error_still_finalizes_restored_interaction(
    monkeypatch,
):
    interaction = FakeInteraction()
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    loop.tokenizer = FakeTokenizer()
    loop._branch_config = SimpleNamespace(candidates=2, continuations=1)
    loop._branch_snapshot = {
        "agent_data": make_agent_data(interaction),
        "interaction": object(),
        "trigger": "after_tool_error",
        "root_candidate_data": make_root_candidate(
            make_agent_data(interaction)
        ),
        "root_candidate_state": AgentState.INTERACTING,
        "root_action_token_ids": [99],
        "root_action_span": [1, 2],
        "prefix": {
            "assistant_turns": 1,
            "user_turns": 1,
            "response_tokens": 1,
            "policy_tokens": 1,
            "tool_calls": 0,
            "message_count": 1,
        },
    }
    loop._branch_task_id = 7
    loop._branch_extra_info = {"split": "train"}
    loop._branch_trajectory = {"step": 0, "rollout_n": 0}
    loop._branch_anchor_turn = 2

    async def fake_generate(self, data, sampling_params, ignore_termination=False):
        data.response_ids = [100]
        data.response_mask.append(1)
        data.tool_calls = []
        return AgentState.INTERACTING

    async def failing_continue(self, data, initial_state, sampling_params):
        raise RuntimeError("suffix failed")

    monkeypatch.setattr(TauBenchAgentLoop, "_handle_generating_state", fake_generate)
    loop._continue_branch = MethodType(failing_continue, loop)

    with pytest.raises(RuntimeError, match="suffix failed"):
        await loop._run_local_branches({})

    assert len(interaction.restored) == 1
    assert interaction.finalized == [
        request_id for request_id, _ in interaction.restored
    ]


@pytest.mark.asyncio
async def test_empty_task_ids_selects_all_tasks(monkeypatch, tmp_path):
    branch_config = AttrDict(
        enable=True,
        validation_only=True,
        anchor_turns=[2],
        candidates=4,
        continuations=2,
        task_ids=[],
        output_dir=str(tmp_path),
    )
    loop = object.__new__(LocalBranchDiagnosticAgentLoop)
    loop.config = AttrDict(
        trainer=AttrDict(val_only=True),
        actor_rollout_ref=AttrDict(
            rollout=AttrDict(multi_turn=AttrDict(local_branch=branch_config))
        ),
    )
    recorded = []
    termination = "unknown"

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = SimpleNamespace(request_id="root-trajectory")
        self.eval_termination = termination
        return SimpleNamespace(extra_fields={})

    monkeypatch.setattr(TauBenchAgentLoop, "run", fake_parent_run)
    monkeypatch.setattr(
        "src.envs.local_branch_agent_loop.append_branch_group",
        lambda output_dir, record: recorded.append(record),
    )

    await loop.run(
        {},
        trajectory={"validate": True, "rollout_n": 0},
        extra_info={"task_id": 7, "split": "train"},
    )

    assert recorded[0]["skip_reason"] == "anchor_not_reached"

    termination = "tool_terminated"
    await loop.run(
        {},
        trajectory={"validate": True, "rollout_n": 0},
        extra_info={"task_id": 7, "split": "train"},
    )
    assert recorded[1]["skip_reason"] == "terminal_before_anchor"

    with pytest.raises(ValueError, match="restricted to the train split"):
        await loop.run(
            {},
            trajectory={"validate": True, "rollout_n": 0},
            extra_info={"task_id": 3, "split": "test"},
        )
