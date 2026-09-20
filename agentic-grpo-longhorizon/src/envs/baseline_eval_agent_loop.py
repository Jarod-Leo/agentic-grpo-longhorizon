"""Compatibility name; training and evaluation use the same loop."""

from src.envs.tau_bench_agent_loop import TauBenchAgentLoop

BaselineEvalAgentLoop = TauBenchAgentLoop
