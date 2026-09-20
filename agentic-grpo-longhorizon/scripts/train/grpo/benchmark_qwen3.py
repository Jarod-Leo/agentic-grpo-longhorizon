"""Run bounded, independent one-step trials through the shared training entry."""

import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time


def main() -> None:
    root = Path(os.environ["RUN_ROOT"])
    budgets = [
        int(value)
        for value in os.environ.get("TOKEN_BUDGETS", "32768,24576,20480").split(",")
    ]
    started = time.monotonic()
    for budget in budgets:
        if time.monotonic() - started > 5400:
            break  # Leave room for one final trial within the two-hour allocation.
        run = root / f"tokens-{budget}"
        run.mkdir(exist_ok=False)
        env = os.environ.copy()
        for key in ("RESUME_FROM", "ADAPTER_PATH"):
            env.pop(key, None)
        env.update(
            RUN_DIR=str(run),
            STEPS="1",
            SAMPLES="8",
            EVAL_SPLIT="train",
            QWEN3_CONFIG="qwen3_unpad",
        )
        with (run / "gpu.csv").open("w") as stream:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,nounits",
                    "-l",
                    "1",
                ],
                stdout=stream,
            )
            began = time.monotonic()
            process = subprocess.Popen(
                [
                    "bash",
                    "scripts/train/grpo/run_qwen3.sh",
                    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={budget}",
                ],
                env=env,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=1500)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc = 124
            finally:
                monitor.terminate()
                monitor.wait()
        (run / "benchmark.json").write_text(
            json.dumps(
                {
                    "token_budget": budget,
                    "exit_code": rc,
                    "wall_seconds": time.monotonic() - began,
                },
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            ["python", "scripts/eval/summarize_qwen3_performance.py", str(root)],
            check=True,
        )
        results = json.loads((root / "performance.json").read_text())
        trial = results["trials"][-1]
        if trial["accepted"] and os.environ.get("CLEAN_BENCHMARK_CHECKPOINTS") == "1":
            # Only exact files inside the newly created candidate directory.
            actor = run / "checkpoints/global_step_1/actor"
            removed = []
            for kind in ("model", "optim", "extra_state"):
                checkpoint = actor / f"{kind}_world_size_1_rank_0.pt"
                removed.append(
                    {"path": str(checkpoint), "bytes": checkpoint.stat().st_size}
                )
                checkpoint.unlink()
            (run / "checkpoint_cleanup.json").write_text(
                json.dumps(removed, indent=2) + "\n"
            )
        if trial["accepted"]:
            break
        log = (
            (run / "run.log").read_text(errors="replace")
            if (run / "run.log").exists()
            else ""
        )
        gpu_oom = bool(
            re.search(
                r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory",
                log,
            )
        )
        if rc == 124 or not gpu_oom:
            break  # API, correctness, CPU-memory and timeout failures are not GPU OOM.
        print(
            f"GPU OOM at token budget {budget}; trying the next lower budget",
            flush=True,
        )
    if not results["recommended_token_budget"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
