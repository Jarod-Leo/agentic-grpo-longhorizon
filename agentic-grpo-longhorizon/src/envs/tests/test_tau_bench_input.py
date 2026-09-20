"""Exercise the real pending hook and per-trajectory reset/cleanup with local fakes."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf
from src.envs.tau_bench_agent_loop import TauBenchAgentLoop
from src.envs.tau_bench_context import CURRENT_TAU_ENV
from src.envs.tau_bench_interaction import TauBenchInteraction
from tau_bench.envs.airline.wiki import WIKI
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState


class Tokenizer:
    def apply_chat_template(self, messages, tools, **kwargs):
        return list(
            (
                json.dumps(tools)
                + "\n"
                + "\n".join(message["content"] for message in messages)
                + "<think>\n\n</think>\n\n"
            ).encode()
        )

    def decode(self, tokens):
        return bytes(tokens).decode()


@pytest.mark.asyncio
async def test_initial_user_once_policy_present_and_context_isolated(monkeypatch):
    import tau_bench.envs

    reset_calls = []

    class Env:
        def reset(self, task_index):
            reset_calls.append(task_index)
            self.task = SimpleNamespace(
                instruction="HIDDEN ANSWER", actions=["HIDDEN ACTION"]
            )
            return SimpleNamespace(observation=f"User request {task_index}")

    monkeypatch.setattr(tau_bench.envs, "get_env", lambda **kwargs: Env())
    interaction = TauBenchInteraction({"reward_mode": "binary"})
    root = Path(__file__).resolve().parents[3]
    source = root / "configs/tool_config/tau_bench_airline_tools.yaml"
    schemas = [
        item["tool_schema"] for item in yaml.safe_load(source.read_text())["tools"]
    ]

    async def one(task_id):
        request_id = str(task_id)
        await interaction.start_interaction(request_id, task_id=task_id)
        env = CURRENT_TAU_ENV.get()
        await asyncio.sleep(0)
        assert CURRENT_TAU_ENV.get() is env
        loop = object.__new__(TauBenchAgentLoop)
        loop.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {"multi_turn": {"tool_config_path": str(source)}}
                }
            }
        )
        loop.tool_schemas = schemas
        loop.tokenizer, loop.processor = Tokenizer(), None
        loop.loop = asyncio.get_running_loop()
        loop.apply_chat_template_kwargs = {"enable_thinking": False}
        loop.prompt_length = 100000
        data = AgentData(
            messages=[{"role": "system", "content": WIKI}],
            image_data=None,
            metrics={},
            request_id=request_id,
            tools_kwargs={},
            interaction=interaction,
            interaction_kwargs={},
        )
        assert await loop._handle_pending_state(data, {}) == AgentState.GENERATING
        rendered = loop.tokenizer.decode(data.prompt_ids)
        assert rendered.count(f"User request {task_id}") == 1
        assert WIKI in rendered and "HIDDEN" not in rendered
        assert data.response_mask == []
        with pytest.raises(AssertionError, match="Expected one policy message"):
            await loop._handle_pending_state(data, {})
        await interaction.finalize_interaction(request_id)
        assert CURRENT_TAU_ENV.get() is None
        return data.messages

    outputs = await asyncio.gather(one(1), one(2))
    assert sorted(reset_calls) == [1, 2]
    assert outputs[0][-1] != outputs[1][-1]
    assert interaction._instance_dict == {}


@pytest.mark.asyncio
async def test_pending_error_releases_environment(monkeypatch):
    from src.envs.audited_tool_agent_loop import AuditedToolAgentLoop

    finalized = []

    class Interaction:
        async def finalize_interaction(self, request_id):
            finalized.append(request_id)

    async def failing_parent(self, sampling_params, **kwargs):
        self.eval_data = SimpleNamespace(request_id="broken", interaction=Interaction())
        raise RuntimeError("render failed")

    monkeypatch.setattr(AuditedToolAgentLoop, "run", failing_parent)
    loop = object.__new__(TauBenchAgentLoop)
    with pytest.raises(RuntimeError, match="render failed"):
        await loop.run({})
    assert finalized == ["broken"]


@pytest.mark.asyncio
async def test_followup_user_and_tool_tokens_are_masked():
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    class FollowupTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [11, 12, 13]

    class Interaction:
        async def generate_response(self, *args, **kwargs):
            return False, "Follow-up request", None, {}

    loop = object.__new__(TauBenchAgentLoop)
    loop.tokenizer, loop.processor = FollowupTokenizer(), None
    loop.loop = asyncio.get_running_loop()
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.system_prompt = []
    loop.max_parallel_calls = 1
    loop.tool_parser_name = "hermes"
    loop.response_length = 100

    async def call_tool(*args):
        return (
            SimpleNamespace(text="Tool observation", image=None, video=None),
            None,
            {},
        )

    loop._call_tool = call_tool
    data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="mask",
        tools_kwargs={},
        interaction=Interaction(),
        interaction_kwargs={},
    )
    data.prompt_ids, data.response_mask = [1, 2], [1, 1]
    await ToolAgentLoop._handle_interacting_state(loop, data)
    assert data.response_mask == [1, 1, 0, 0, 0]
    data.tool_calls = [SimpleNamespace(name="get_user_details")]
    await ToolAgentLoop._handle_processing_tools_state(loop, data)
    assert data.response_mask == [1, 1, 0, 0, 0, 0, 0, 0]
