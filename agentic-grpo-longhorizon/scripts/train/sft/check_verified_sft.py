"""Cheap gates for verified token masks and the exported PEFT adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer


def check_data(path: Path, model: str, min_samples: int) -> None:
    frame = pd.read_parquet(path)
    if len(frame) < min_samples:
        raise ValueError(f"Verified SFT requires at least {min_samples} samples, found {len(frame)}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Qwen3 non-thinking generation prompt changed")

    for _, row in frame.iterrows():
        input_ids = np.asarray(row["input_ids"], dtype=np.int64)
        loss_mask = np.asarray(row["loss_mask"], dtype=np.int64)
        loss_start = int(row["loss_start"])
        if len(input_ids) != len(loss_mask) or not loss_mask.any():
            raise ValueError("Invalid token/mask lengths")
        if loss_mask[:loss_start].any():
            raise ValueError("Student prefix contributes to SFT loss")
        target = tokenizer.decode(input_ids[loss_mask.astype(bool)], skip_special_tokens=False)
        if any(value.strip() for value in re.findall(r"<think>(.*?)</think>", target, re.DOTALL)):
            raise ValueError("Teacher target contains non-empty thinking")
    print(f"VERIFIED_SFT_DATA_OK rows={len(frame)} path={path}")


def check_adapter(path: Path) -> None:
    config = json.loads((path / "adapter_config.json").read_text())
    if int(config["r"]) != 16 or int(config["lora_alpha"]) != 32:
        raise ValueError("Unexpected LoRA rank/alpha")
    weights = path / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
    if not keys or any("lora_" not in key for key in keys):
        raise ValueError("Adapter contains no LoRA weights")
    print(f"VERIFIED_SFT_ADAPTER_OK tensors={len(keys)} path={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.data:
        if not args.model:
            parser.error("--model is required with --data")
        check_data(args.data, args.model, args.min_samples)
    if args.adapter:
        check_adapter(args.adapter)
    if not args.data and not args.adapter:
        parser.error("Provide --data and/or --adapter")


if __name__ == "__main__":
    main()
