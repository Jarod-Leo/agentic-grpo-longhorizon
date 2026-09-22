"""Freeze shared runtime sources before sbatch; outputs remain outside the snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    sources = [
        "agentic-grpo-longhorizon/src",
        "agentic-grpo-longhorizon/scripts",
        "agentic-grpo-longhorizon/configs",
        "verl/verl",
        "tau-bench/tau_bench",
        "agentic-grpo-longhorizon/experiments/e00_qwen3_baseline/split.json",
        "verl/tests/trainer/ppo/test_grpo_signal_on_cpu.py",
        "verl/tests/trainer/ppo/test_grpo_lata_on_cpu.py",
        "verl/tests/trainer/ppo/test_distillation_loss_on_cpu.py",
    ]
    manifest = {}
    for source in sources:
        path = root / source
        files = path.rglob("*") if path.is_dir() else [path]
        for file in files:
            if (
                not file.is_file()
                or "__pycache__" in file.parts
                or file.suffix in {".orig", ".parquet", ".pt", ".pyc", ".rej"}
            ):
                continue
            relative = file.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            manifest[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
    relative = Path(
        "agentic-grpo-longhorizon/experiments/sft_collect_airline/split.json"
    )
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative, target)
    manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Frozen {len(manifest)} files at {destination}")


if __name__ == "__main__":
    main()
