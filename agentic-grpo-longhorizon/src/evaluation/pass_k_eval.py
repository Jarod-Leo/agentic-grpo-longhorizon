"""
重复轨迹评测。

pass@k 表示 k 次中至少一次成功，pass^k 表示 k 次全部成功。
同时统计 turn efficiency 和 tool call accuracy。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tqdm import tqdm

from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult
from src.evaluation.pass_metrics import _mean_available, task_pass_metrics


def _get_tokenizer(policy_factory):
    """尝试从 policy 获取 model_name 并加载 tokenizer；失败则返回 None。"""
    try:
        policy = policy_factory()
        model_name = getattr(policy, "model_name", None)
        if model_name:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"[pass_k_eval] Loaded tokenizer for {model_name}")
            return tok
    except Exception as e:
        print(f"[pass_k_eval] Failed to load tokenizer: {e}, falling back to char length")
    return None


def _make_token_counter(tokenizer):
    """返回一个 callable: text -> token count"""
    if tokenizer is not None:

        def _count(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                return len(text)

        return _count
    return len


@dataclass
class EvalReport:
    metric_schema_version: int
    env_name: str
    num_tasks: int
    num_samples_per_task: int
    pass_at_1: float | None  # 单次尝试的平均成功率
    pass_at_4: float | None  # 4 次中至少一次成功
    pass_at_8: float | None  # 8 次中至少一次成功
    any_success_rate: float | None  # 每个 task 至少一次成功的比例
    pass_hat_1: float | None  # 兼容字段：pass^1 = 平均成功率
    pass_hat_4: float | None  # 兼容字段：pass^4 = 4 次全部成功
    pass_hat_8: float | None  # 兼容字段：pass^8 = 8 次全部成功
    avg_turns: float
    avg_tool_calls: float
    error_rate: float          # trajectory 异常中止的比例
    per_task_results: list[dict]


def run_eval(
    wrapper: TauBenchWrapper,
    policy_factory,                    # callable -> policy instance (thread-safe)
    num_tasks: int | None = None,
    num_samples_per_task: int = 4,
    max_turns: int = 30,
    num_workers: int = 4,
    output_dir: str = "experiments/baseline_airline_7B_user",
) -> EvalReport:
    """
    policy_factory: 每个 worker 线程自己 new 一个 policy,避免并发问题
    """
    if num_tasks is None:
        num_tasks = wrapper.get_num_tasks()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每个 (task_idx, sample_idx) 是一个独立 job
    jobs = [(t, s) for t in range(num_tasks) for s in range(num_samples_per_task)]
    results: dict[int, list[TrajectoryResult]] = {t: [] for t in range(num_tasks)}

    def _run_one(task_idx: int, sample_idx: int) -> tuple[int, TrajectoryResult]:
        policy = policy_factory()
        # 给同一个 task 不同 sample 设不同 temperature seed
        # (vLLM server 端已经有采样随机性,这里主要是逻辑标记)
        traj = wrapper.run_single_task(task_idx, policy, max_turns=max_turns)
        return task_idx, traj

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_one, t, s) for t, s in jobs]
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Eval"):
            task_idx, traj = fut.result()
            results[task_idx].append(traj)

    import numpy as np

    # 尝试加载 tokenizer（用于精确统计 assistant content tokens）
    tokenizer = _get_tokenizer(policy_factory)
    count_tokens = _make_token_counter(tokenizer)

    per_task = []
    success_rate_list, pass_at_4_list, pass_at_8_list = [], [], []
    pass_all_4_list, pass_all_8_list = [], []
    any_success_list = []
    all_turns, all_tool_calls, all_errors = [], [], []

    for t in range(num_tasks):
        trajs = results[t]
        n = len(trajs)
        c = sum(1 for tr in trajs if tr.success)

        metrics = task_pass_metrics(n, c)
        success_rate = metrics["success_rate"]
        pass_at = metrics["pass_at_k"]
        pass_all = metrics["pass_all_k"]

        success_rate_list.append(success_rate)
        pass_at_4_list.append(pass_at[4])
        pass_at_8_list.append(pass_at[8])
        pass_all_4_list.append(pass_all[4])
        pass_all_8_list.append(pass_all[8])
        any_success_list.append(None if n == 0 else 1.0 if c > 0 else 0.0)

        traj_dicts = []
        for tr in trajs:
            all_turns.append(tr.num_turns)
            all_tool_calls.append(tr.num_tool_calls)
            all_errors.append(1.0 if tr.error else 0.0)

            # 计算每轮 assistant turn 的 content token 数
            per_turn_assistant_content_tokens = []
            for msg in tr.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "") or ""
                    per_turn_assistant_content_tokens.append(count_tokens(content))

            traj_dict = tr.to_dict()
            traj_dict["per_turn_assistant_content_tokens"] = per_turn_assistant_content_tokens
            traj_dicts.append(traj_dict)

        per_task.append({
            "task_id": t,
            "success_count": c,
            "total_samples": n,
            "success_rate": success_rate,
            "pass@1": pass_at[1],
            "pass@4": pass_at[4],
            "pass@8": pass_at[8],
            "pass^1": pass_all[1],
            "pass^4": pass_all[4],
            "pass^8": pass_all[8],
            "any_success": c > 0,
            "avg_turns": np.mean([tr.num_turns for tr in trajs]),
            "trajectories": traj_dicts,
        })

    pass_at_1 = _mean_available(success_rate_list)
    report = EvalReport(
        metric_schema_version=2,
        env_name=wrapper.env_name,
        num_tasks=num_tasks,
        num_samples_per_task=num_samples_per_task,
        pass_at_1=pass_at_1,
        pass_at_4=_mean_available(pass_at_4_list),
        pass_at_8=_mean_available(pass_at_8_list),
        any_success_rate=_mean_available(any_success_list),
        pass_hat_1=pass_at_1,
        pass_hat_4=_mean_available(pass_all_4_list),
        pass_hat_8=_mean_available(pass_all_8_list),
        avg_turns=float(np.mean(all_turns)),
        avg_tool_calls=float(np.mean(all_tool_calls)),
        error_rate=float(np.mean(all_errors)),
        per_task_results=per_task,
    )

    # 保存
    with open(output_dir / "eval_report.json", "w") as f:
        #json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

    # 打印摘要
    print(f"\n=== Eval Report: {wrapper.env_name} ===")
    print(f"Tasks: {num_tasks} × Samples: {num_samples_per_task}")
    def _format_metric(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "N/A"

    print(f"pass@1 (single-attempt mean): {_format_metric(report.pass_at_1)}")
    print(f"pass@4 (at least one):        {_format_metric(report.pass_at_4)}")
    print(f"pass@8 (at least one):        {_format_metric(report.pass_at_8)}")
    print(f"any-success task rate:        {_format_metric(report.any_success_rate)}")
    print(f"pass^4 (all successful):      {_format_metric(report.pass_hat_4)}")
    print(f"pass^8 (all successful):      {_format_metric(report.pass_hat_8)}")
    print(f"Avg turns:       {report.avg_turns:.2f}")
    print(f"Avg tool calls:  {report.avg_tool_calls:.2f}")
    print(f"Error rate:      {report.error_rate:.3f}")

    return report
