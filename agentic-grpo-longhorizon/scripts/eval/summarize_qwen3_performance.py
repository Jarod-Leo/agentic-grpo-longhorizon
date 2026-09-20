"""Summarize one-step timing and device-wide memory for Qwen3 performance trials."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(root: Path) -> dict:
    trials = []
    for path in sorted(root.glob("tokens-*/benchmark.json"), reverse=True):
        run = path.parent
        trial = json.loads(path.read_text())
        summary = (
            json.loads((run / "summary.json").read_text())
            if (run / "summary.json").exists()
            else {}
        )
        metrics = (
            [
                json.loads(line)
                for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            if (run / "metrics.jsonl").exists()
            else []
        )
        data = metrics[-1]["data"] if metrics else {}
        with (run / "gpu.csv").open() as stream:
            samples = list(csv.DictReader(stream, skipinitialspace=True))
        fractions = [
            float(row["memory.used [MiB]"]) / float(row["memory.total [MiB]"])
            for row in samples
        ]
        peak = max(fractions) if fractions else None
        grad = data.get("actor/grad_norm")
        signal = summary.get("learning", {}).get("signal_observed", False)
        trial.update(
            accepted=bool(
                trial["exit_code"] == 0
                and summary.get("accepted")
                and len(metrics) == 1
                and grad is not None
                and math.isfinite(grad)
                and (not signal or grad > 0)
                and peak is not None
            ),
            peak_device_memory_fraction=peak,
            grad_norm=grad,
            signal_observed=signal,
            gpu_samples=len(samples),
            rollout_seconds=data.get("timing_s/gen"),
            actor_update_seconds=data.get("timing_s/update_actor"),
            step_seconds=data.get("timing_s/step"),
            checkpoint_seconds=data.get("timing_s/save_checkpoint"),
            effective_tokens=data.get("perf/total_num_tokens"),
            torch_peak_allocated_gib=data.get("perf/max_memory_allocated_gb"),
            torch_peak_reserved_gib=data.get("perf/max_memory_reserved_gb"),
        )
        if (
            trial["step_seconds"] is not None
            and trial["checkpoint_seconds"] is not None
        ):
            trial["step_with_checkpoint_seconds"] = (
                trial["step_seconds"] + trial["checkpoint_seconds"]
            )
            trial["startup_and_other_seconds"] = (
                trial["wall_seconds"] - trial["step_with_checkpoint_seconds"]
            )
        trials.append(trial)
    eligible = [trial for trial in trials if trial["accepted"]]
    best = (
        min(eligible, key=lambda trial: trial["actor_update_seconds"])
        if eligible
        else None
    )
    return {
        "trials": trials,
        "recommended_token_budget": best["token_budget"] if best else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    result = summarize(root)
    (root / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Qwen3 dynamic microbatch performance",
        "",
        "Single-step measurements; rollout lengths may differ. Startup is excluded from step time.",
        "",
        "| Token budget | Accepted | GPU peak % | Rollout s | Actor s | Step s | Save s | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for trial in result["trials"]:
        peak = trial["peak_device_memory_fraction"]
        values = [
            trial["token_budget"],
            trial["accepted"],
            round(100 * peak, 1) if peak is not None else None,
            trial["rollout_seconds"],
            trial["actor_update_seconds"],
            trial["step_seconds"],
            trial["checkpoint_seconds"],
            trial["effective_tokens"],
        ]
        lines.append(
            "| "
            + " | ".join(
                str(round(v, 2)) if isinstance(v, float) else str(v) for v in values
            )
            + " |"
        )
    lines.extend(
        ["", f"Recommended token budget: {result['recommended_token_budget']}", ""]
    )
    (root / "performance.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
