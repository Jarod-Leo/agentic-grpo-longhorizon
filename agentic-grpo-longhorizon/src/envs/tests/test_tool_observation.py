"""Regression coverage for the observation delivered by veRL's real tool path."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse


class ReservationTool:
    def __init__(self, response: str):
        self.response = response
        self.released = []

    async def create(self, **kwargs):
        return "reservation-instance", ToolResponse()

    async def execute(self, instance_id, parameters):
        assert instance_id == "reservation-instance"
        assert parameters == {"reservation_id": "ABC123"}
        return ToolResponse(text=self.response), 0.0, {}

    async def release(self, instance_id):
        self.released.append(instance_id)


async def call_reservation_tool(response: str, max_length: int, truncate_side: str):
    tool = ReservationTool(response)
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"get_reservation_details": tool}
    loop.max_tool_response_length = max_length
    loop.tool_response_truncate_side = truncate_side
    result, reward, metadata = await ToolAgentLoop._call_tool(
        loop,
        FunctionCall(
            name="get_reservation_details",
            arguments=json.dumps({"reservation_id": "ABC123"}),
        ),
        {},
    )
    assert tool.released == ["reservation-instance"]
    assert reward == 0.0 and metadata == {}
    return result.text


@pytest.mark.asyncio
async def test_formal_config_preserves_complete_reservation_observation(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("REPO", str(root))
    monkeypatch.setenv("RUN_DIR", "/tmp/tool-observation-config")
    monkeypatch.setenv("EXP_NAME", "tool_observation_config")
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/tool-observation-ray")
    monkeypatch.setenv("POLICY_MODEL", "/tmp/qwen3-model")
    with initialize_config_dir(
        config_dir=str(root / "configs/train/grpo"), version_base=None
    ):
        config = compose(config_name="qwen3_formal")

    response = json.dumps(
        {
            "reservation_id": "ABC123",
            "passenger_notes": "p" * 300,
            "payment_history": [
                {"payment_id": "credit_card_7815826", "amount": 426.75}
            ],
            "flights": [
                {"flight_number": "HAT789", "date": "2026-11-02"}
            ],
            "policy_notes": "s" * 300,
        }
    )
    configured_limit = int(
        config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
    )
    assert 256 < len(response) < configured_limit

    old_observation = await call_reservation_tool(response, 256, "middle")
    assert "...(truncated)..." in old_observation
    assert "credit_card_7815826" not in old_observation
    assert "HAT789" not in old_observation

    multi_turn = config.actor_rollout_ref.rollout.multi_turn
    observation = await call_reservation_tool(
        response,
        configured_limit,
        str(multi_turn.tool_response_truncate_side),
    )
    assert observation == response
    assert (
        json.loads(observation)["payment_history"][0]["payment_id"]
        == "credit_card_7815826"
    )
    assert json.loads(observation)["flights"][0]["flight_number"] == "HAT789"
