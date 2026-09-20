"""Optional readiness diagnostics using the same tool loop and execution behavior."""

import json
import os
import time

from src.envs.tau_bench_context import CURRENT_TAU_STATE
from src.envs.tool_call_audit import inspect_tool_output
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


class AuditedToolAgentLoop(ToolAgentLoop):
    def _write_audit(self, event):
        path = os.environ["TOOL_AUDIT_PATH"]
        state = CURRENT_TAU_STATE.get() or {}
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "trajectory_id": self.audit_request_id,
                        "task_id": state.get("task_id"),
                        **event,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    async def _handle_generating_state(
        self, agent_data, sampling_params, ignore_termination=False
    ):
        self.audit_request_id = agent_data.request_id
        suffix = self.tokenizer.decode(agent_data.prompt_ids[-32:])
        # Check every actual generation prompt, including tool/user continuation turns.
        if self.apply_chat_template_kwargs.get("enable_thinking") is False:
            assert suffix.endswith("<think>\n\n</think>\n\n"), (
                "Non-thinking prompt suffix was lost"
            )
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        self._write_audit(
            {
                "event": "generation",
                "turn": agent_data.assistant_turns,
                "text": text,
                "tokens": len(agent_data.response_ids),
                "terminated_at_limit": state == AgentState.TERMINATED,
                **inspect_tool_output(text, self.tool_schemas),
            }
        )
        return state

    async def _call_tool(self, tool_call, tools_kwargs):
        response = await super()._call_tool(tool_call, tools_kwargs)
        text = response[0].text or ""
        self._write_audit(
            {
                "event": "execution",
                "name": tool_call.name,
                "error": text.lower().startswith(("error", "unknown action")),
                "response": text,
            }
        )
        return response
