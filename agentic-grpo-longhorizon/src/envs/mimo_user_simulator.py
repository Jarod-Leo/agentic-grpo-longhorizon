"""MiMo user history with explicit, task-aware initialization."""

from src.envs.mimo_client import estimated_cost_usd
from tau_bench.envs.user import LLMUserSimulationEnv


class MimoUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, client, trajectory_id):
        self.client = client
        self.trajectory_id = trajectory_id
        self.messages = []
        self.total_cost = 0.0
        # Env.reset supplies the actual task; do not generate a discarded greeting.

    def generate_next_message(self, messages):
        content, usage = self.client.complete(messages, self.trajectory_id)
        self.messages.append({"role": "assistant", "content": content})
        self.total_cost += estimated_cost_usd(usage)
        return content
