"""Grounding and lifecycle tests for Agent-OPSD F2 feedback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray")

from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop
from src.envs.opsd_feedback import (
    ContaminatedTrajectoryError,
    OPSD_FEEDBACK_VERSION,
    OpsdFeedbackError,
    build_opsd_feedback,
    empty_opsd_feedback,
)
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopWorkerBase,
    _InternalAgentLoopOutput,
)


def action(tool: str, parameters: dict, *, is_error: bool, **private_fields) -> dict:
    return {
        "tool": tool,
        "parameters": parameters,
        "is_error": is_error,
        "inc_reward": private_fields.get("inc_reward", 0.0),
        "done": private_fields.get("done", False),
        "content": private_fields.get("content", ""),
        "extracted_entities": private_fields.get("extracted_entities", {}),
    }


def state(*actions: dict, contaminated: bool = False) -> dict:
    return {"contaminated": contaminated, "action_history": list(actions)}


def test_f2_uses_only_grounded_error_and_exact_repeat_signals() -> None:
    secret = "ZXCVBN"
    feedback = build_opsd_feedback(
        state(
            action(
                "get_reservation_details",
                {"reservation_id": secret},
                is_error=True,
                content="future user dialogue",
                extracted_entities={"reservation_id": [secret]},
            ),
            action(
                "get_reservation_details",
                {"reservation_id": secret},
                is_error=True,
            ),
            action(
                "get_reservation_details",
                {"reservation_id": "DIFFERENT"},
                is_error=True,
            ),
            action("search_direct_flight", {"origin": "SIN"}, is_error=False),
        ),
        outcome_reward=0.0,
        termination="assistant_turn_limit",
    )

    assert set(feedback) == {"version", "mode", "text", "rules"}
    assert feedback["version"] == "opsd-f2-v1"
    assert feedback["mode"] == "F2"
    assert feedback["rules"] == [
        "terminal_outcome",
        "termination_reason",
        "tool_error",
        "repeated_failed_call",
    ]
    assert "Terminal outcome: failure" in feedback["text"]
    assert (
        "same failed get_reservation_details call was repeated 2 times"
        in feedback["text"]
    )
    serialized = json.dumps(feedback, ensure_ascii=False)
    assert secret not in serialized
    assert "DIFFERENT" not in serialized
    assert "future user dialogue" not in serialized
    assert "reservation_id" not in serialized
    assert "search_direct_flight" not in serialized


def test_f2_does_not_infer_errors_or_repeat_from_reward() -> None:
    feedback = build_opsd_feedback(
        state(
            action(
                "search_direct_flight",
                {"origin": "SIN"},
                is_error=False,
                inc_reward=0.0,
                done=True,
            ),
            action(
                "search_direct_flight",
                {"origin": "SIN"},
                is_error=False,
                inc_reward=-1.0,
                done=True,
            ),
        ),
        outcome_reward=1,
        termination="interaction_terminated",
    )

    assert feedback["rules"] == ["terminal_outcome", "termination_reason"]
    assert "Tool error" not in feedback["text"]
    assert "repeated" not in feedback["text"]


def test_failed_parameter_comparison_is_exact_and_case_sensitive() -> None:
    feedback = build_opsd_feedback(
        state(
            action("get_user_details", {"user_id": "Case_1"}, is_error=True),
            action("get_user_details", {"user_id": "case_1"}, is_error=True),
        ),
        outcome_reward=0,
        termination="response_budget",
    )

    assert "tool_error" in feedback["rules"]
    assert "repeated_failed_call" not in feedback["rules"]


def test_contaminated_or_ungrounded_state_is_rejected() -> None:
    with pytest.raises(ContaminatedTrajectoryError, match="contaminated"):
        build_opsd_feedback(
            state(contaminated=True),
            outcome_reward=0,
            termination="interaction_terminated",
        )
    with pytest.raises(OpsdFeedbackError, match="is_error"):
        build_opsd_feedback(
            {"contaminated": False, "action_history": [{"tool": "think"}]},
            outcome_reward=0,
            termination="interaction_terminated",
        )
    with pytest.raises(OpsdFeedbackError, match="terminal state"):
        build_opsd_feedback(state(), outcome_reward=0, termination="secret reason")
    with pytest.raises(OpsdFeedbackError, match="version"):
        build_opsd_feedback(
            state(),
            outcome_reward=0,
            termination="interaction_terminated",
            version="unreviewed",
        )


def test_f0_is_an_explicit_empty_negative_control() -> None:
    assert empty_opsd_feedback() == {
        "version": "opsd-f2-v1",
        "mode": "F0",
        "text": "",
        "rules": [],
    }


class FakeInteraction:
    reward_mode = "binary"

    def __init__(self, trajectory_state: dict) -> None:
        self.trajectory_state = trajectory_state
        self.state_reads = 0
        self.finalized = []

    def opsd_feedback_state(self, request_id: str) -> dict:
        assert request_id not in self.finalized
        self.state_reads += 1
        return self.trajectory_state

    async def finalize_interaction(self, request_id: str) -> None:
        self.finalized.append(request_id)


def make_loop(
    feedback_mode: str = "F2", teacher_mode: str = "self_feedback"
) -> TauBenchAgentLoop:
    loop = object.__new__(TauBenchAgentLoop)
    loop.config = {
        "tau_bench_reward_mode": "binary",
        "distillation": {
            "teacher_mode": teacher_mode,
            "feedback_mode": feedback_mode,
            "feedback_version": OPSD_FEEDBACK_VERSION,
        },
    }
    loop.tokenizer = SimpleNamespace(
        decode=lambda token_ids: "|".join(map(str, token_ids))
    )
    return loop


def make_data(interaction: FakeInteraction) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="opsd-trajectory",
        assistant_turns=2,
        user_turns=1,
        total_tool_calls=2,
        messages=[],
        interaction=interaction,
    )


def make_output() -> SimpleNamespace:
    return SimpleNamespace(
        reward_score=0.0,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 0],
        extra_fields={"outcome_reward": 0.0, "process_score": 0.0},
    )


async def run_loop(
    monkeypatch,
    tmp_path,
    interaction: FakeInteraction,
    loop: TauBenchAgentLoop,
    *,
    validate: bool = False,
):
    output = make_output()
    data = make_data(interaction)

    async def fake_parent_run(self, sampling_params, **kwargs):
        self.eval_data = data
        self.eval_termination = "interaction_terminated"
        return output

    monkeypatch.setattr(AuditedToolAgentLoop, "run", fake_parent_run)
    monkeypatch.setenv("TRAJECTORIES_PATH", str(tmp_path / "trajectories.jsonl"))
    result = await loop.run(
        {},
        trajectory={"step": 1, "validate": validate},
        extra_info={"protocol": "opsd", "task_id": 3, "split": "train"},
    )
    return result


@pytest.mark.asyncio
async def test_loop_captures_feedback_before_finalize(monkeypatch, tmp_path) -> None:
    interaction = FakeInteraction(
        state(action("get_user_details", {"user_id": "private_1"}, is_error=True))
    )
    result = await run_loop(monkeypatch, tmp_path, interaction, make_loop())

    assert result.extra_fields["opsd_feedback"]["mode"] == "F2"
    assert interaction.state_reads == 1
    assert interaction.finalized == ["opsd-trajectory"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feedback_mode", "teacher_mode"),
    [("F0", "self_feedback"), ("F2", "external")],
)
async def test_loop_uses_f0_only_for_self_feedback_training(
    monkeypatch, tmp_path, feedback_mode, teacher_mode
) -> None:
    interaction = FakeInteraction(state())
    result = await run_loop(
        monkeypatch,
        tmp_path,
        interaction,
        make_loop(feedback_mode=feedback_mode, teacher_mode=teacher_mode),
    )

    if teacher_mode == "self_feedback":
        assert result.extra_fields["opsd_feedback"] == empty_opsd_feedback()
    else:
        assert "opsd_feedback" not in result.extra_fields
    assert interaction.state_reads == 0
    assert interaction.finalized == ["opsd-trajectory"]


@pytest.mark.asyncio
async def test_loop_rejects_contamination_and_still_finalizes(
    monkeypatch, tmp_path
) -> None:
    interaction = FakeInteraction(state(contaminated=True))
    with pytest.raises(ContaminatedTrajectoryError, match="contaminated"):
        await run_loop(monkeypatch, tmp_path, interaction, make_loop())
    assert interaction.finalized == ["opsd-trajectory"]
    assert not (tmp_path / "trajectories.jsonl").exists()


@pytest.mark.asyncio
async def test_validation_skips_feedback_and_keeps_binary_result(
    monkeypatch, tmp_path
) -> None:
    interaction = FakeInteraction(state(contaminated=True))
    result = await run_loop(
        monkeypatch, tmp_path, interaction, make_loop(), validate=True
    )

    assert result.reward_score == 0.0
    assert "opsd_feedback" not in result.extra_fields
    assert interaction.state_reads == 0
    assert interaction.finalized == ["opsd-trajectory"]


def test_framework_postprocess_preserves_feedback_object() -> None:
    feedback = build_opsd_feedback(
        state(), outcome_reward=0, termination="interaction_terminated"
    )
    output = _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]]),
        response_ids=torch.tensor([[3, 4]]),
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=torch.tensor([[0, 1, 2, 3]]),
        response_mask=torch.tensor([[1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        reward_score=0.0,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"opsd_feedback": feedback},
    )
    worker = object.__new__(AgentLoopWorkerBase)

    batch = worker._postprocess([output])

    assert batch.non_tensor_batch["opsd_feedback"].dtype == object
    assert batch.non_tensor_batch["opsd_feedback"].tolist() == [feedback]
