"""Build the shared train/test checkpoint comparison from existing summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:.2%}"


def duration(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:.3f} GPU h"


def success_count(row: dict) -> str:
    n = row.get("n")
    count = row.get("c")
    tasks = row.get("tasks")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (n, count)
    ):
        return "—"
    task_text = f"；{tasks} 任务" if isinstance(tasks, int) else ""
    return f"{count}/{n}{task_text}"


def summary_link(report_path: Path, summary_path: Path) -> str:
    target = Path(os.path.relpath(summary_path, report_path.parent)).as_posix()
    return f"[summary]({target})"


def result_row(report_path: Path, model: dict, split: str) -> list[str]:
    summary_path = Path(model[f"{split}_summary"])
    link = summary_link(report_path, summary_path)
    if not summary_path.exists():
        return [model["name"], split, f"待评测（{link}）", "—", "—", "—", "—", "—"]

    try:
        summary = read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [
            model["name"],
            split,
            f"读取失败：{markdown_text(error)}（{link}）",
            "—",
            "—",
            "—",
            "—",
            "—",
        ]

    split_row = summary.get("splits", {}).get(split)
    if not isinstance(split_row, dict):
        return [
            model["name"],
            split,
            f"未验收（缺少 {split} 指标；{link}）",
            "—",
            "—",
            "—",
            "—",
            duration(summary.get("gpu_hours")),
        ]

    terminal = summary.get("terminal_behavior")
    terminal_status = (
        "终止检查通过"
        if isinstance(terminal, dict) and terminal.get("passed") is True
        else "终止检查未通过"
        if isinstance(terminal, dict) and terminal.get("passed") is False
        else "终止检查缺失"
    )
    if summary.get("accepted") is True and terminal_status == "终止检查通过":
        status = f"已验收（{terminal_status}；{link}）"
    elif summary.get("accepted") is True:
        status = f"未验收（accepted=true；{terminal_status}；{link}）"
    elif summary.get("accepted") is False:
        status = f"未验收（accepted=false；{terminal_status}；{link}）"
    else:
        status = f"未验收（accepted 缺失；{link}）"

    return [
        model["name"],
        split,
        status,
        success_count(split_row),
        percent(split_row.get("success_rate")),
        percent(split_row.get("pass_all_4")),
        percent(split_row.get("pass_at_4")),
        duration(summary.get("gpu_hours")),
    ]


def build_report(manifest: dict, report_path: Path) -> str:
    protocol = manifest.get("protocol", {})
    models = manifest.get("models", [])
    if not isinstance(protocol, dict) or not isinstance(models, list):
        raise ValueError("Manifest protocol must be an object and models must be a list")

    lines = [
        "# 修复终止语义后的模型复评",
        "",
        "所有结果直接读取各次评测生成的 `summary.json`；待评测和未验收结果均保留在表中。",
        "",
        "## 统一协议",
        "",
        "| 配置 | 值 |",
        "| --- | --- |",
    ]
    for key, value in protocol.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        lines.append(f"| {markdown_text(key)} | {markdown_text(rendered)} |")

    lines += [
        "",
        "## 结果",
        "",
        "| 模型 | 划分 | 状态 | 成功数 | Pass@1 | Pass^4 | Pass@4 | 时长 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        if not isinstance(model, dict) or "name" not in model:
            raise ValueError("Each model entry must be an object with a name")
        for split in ("train", "test"):
            if f"{split}_summary" not in model:
                raise ValueError(f"Model {model['name']} is missing {split}_summary")
            row = result_row(report_path, model, split)
            lines.append("| " + " | ".join(markdown_text(value) for value in row) + " |")

    lines += ["", "## Checkpoint", ""]
    for model in models:
        lines.append(f"- {markdown_text(model['name'])}: `{model.get('checkpoint', '—')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    report_path = Path(manifest["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(manifest, report_path), encoding="utf-8")
    print(f"QWEN3_SUITE_REPORT path={report_path}")


if __name__ == "__main__":
    main()
