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

from src.envs.mimo_client import MimoClient


_TEACHER_API_SCENARIOS = [
    {
        "name": "single_tool",
        "prompt": "Use get_order once to look up order ORDER-100.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Look up one order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    },
    {
        "name": "independent_tools",
        "prompt": "Look up order ORDER-200 and the electronics refund policy using the provided tools.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Look up one order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_refund_policy",
                    "description": "Look up the refund policy for one product category.",
                    "parameters": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
    },
    {
        "name": "dependent_tools",
        "prompt": (
            "Find external customer CUST-300, then update that customer's address to 3 Research Drive using the "
            "provided tools."
        ),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "find_customer",
                    "description": "Resolve an external customer ID to an internal record ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"external_customer_id": {"type": "string"}},
                        "required": ["external_customer_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_customer_address",
                    "description": "Update an address using the internal record ID returned by find_customer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "internal_record_id": {"type": "string"},
                            "address": {"type": "string"},
                        },
                        "required": ["internal_record_id", "address"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
    },
]


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


def check_teacher_api(model: str, base_url: str) -> None:
    client = MimoClient(
        {
            "user_model": model,
            "user_base_url": base_url,
            "max_completion_tokens": 256,
            "rpm": 100,
            "tpm": 1_000_000,
            "max_inflight": 1,
        }
    )
    records: list[dict] = []
    record = client._record

    def capture_record(row: dict) -> None:
        records.append(row.copy())
        record(row)

    client._record = capture_record
    for scenario in _TEACHER_API_SCENARIOS:
        rows_before = len(records)
        content, _usage = client.complete(
            [{"role": "user", "content": scenario["prompt"]}],
            f"teacher-api-preflight:{scenario['name']}",
            tools=scenario["tools"],
            role="teacher",
        )
        scenario_rows = records[rows_before:]
        if not scenario_rows or not scenario_rows[-1]["success"] or scenario_rows[-1]["tool_calls"] != 1:
            raise ValueError(f"Teacher API scenario did not return exactly one native tool call: {scenario['name']}")
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        if len(blocks) != 1:
            raise ValueError(f"Teacher API scenario returned an invalid rendered tool call: {scenario['name']}")
        payload = json.loads(blocks[0])
        allowed_names = {tool["function"]["name"] for tool in scenario["tools"]}
        if payload.get("name") not in allowed_names or not isinstance(payload.get("arguments"), dict):
            raise ValueError(f"Teacher API scenario returned an invalid tool payload: {scenario['name']}")
        print(
            f"VERIFIED_SFT_TEACHER_API_SCENARIO_OK name={scenario['name']} tool={payload['name']} "
            f"attempts={len(scenario_rows)}"
        )
    print(f"VERIFIED_SFT_TEACHER_API_OK scenarios={len(_TEACHER_API_SCENARIOS)} model={model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--teacher-api", action="store_true")
    parser.add_argument("--teacher-model", default="mimo-v2.6-flash")
    parser.add_argument("--teacher-base-url", default="https://api.xiaomimimo.com/v1")
    args = parser.parse_args()
    if args.data:
        if not args.model:
            parser.error("--model is required with --data")
        check_data(args.data, args.model, args.min_samples)
    if args.adapter:
        check_adapter(args.adapter)
    if args.teacher_api:
        check_teacher_api(args.teacher_model, args.teacher_base_url)
    if not args.data and not args.adapter and not args.teacher_api:
        parser.error("Provide --data, --adapter and/or --teacher-api")


if __name__ == "__main__":
    main()
