"""Strict tool-call diagnostics, independent of permissive runtime repair."""

import json
import re

from jsonschema import Draft7Validator


def inspect_tool_output(text: str, schemas: list[dict]) -> dict:
    """Measure strict JSON/schema validity, separately from permissive parser repair."""
    schema_map = {s["function"]["name"]: s["function"]["parameters"] for s in schemas}
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    calls = []
    for block in blocks:
        item = {"json_valid": False, "known_name": False, "schema_valid": False}
        try:
            call = json.loads(block)
            item["json_valid"] = (
                isinstance(call, dict)
                and isinstance(call.get("name"), str)
                and isinstance(call.get("arguments"), dict)
            )
            if item["json_valid"]:
                item["name"] = call["name"]
                item["known_name"] = call["name"] in schema_map
                if item["known_name"]:
                    item["schema_valid"] = Draft7Validator(
                        schema_map[call["name"]]
                    ).is_valid(call["arguments"])
        except (ValueError, TypeError):
            pass
        calls.append(item)
    return {
        "tool_tag_starts": text.count("<tool_call>"),
        "tool_tag_ends": text.count("</tool_call>"),
        "closed_blocks": len(blocks),
        "calls": calls,
        "nonempty_thinking": any(
            x.strip() for x in re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        ),
    }
