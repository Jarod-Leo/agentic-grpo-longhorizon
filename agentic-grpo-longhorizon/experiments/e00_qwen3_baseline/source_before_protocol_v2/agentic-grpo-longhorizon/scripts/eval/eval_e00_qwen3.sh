#!/bin/bash
# =============================================================================
# E00：官方 Qwen3-8B 能力基线评测（veRL 进程内 val_only，40 题 × 8 = 320 条轨迹）
#
# 集群入口是 cluster_setup/e00_qwen3_baseline/eval.sbatch，它负责 #SBATCH、
# module/activate.sh 与 triton shim。本脚本只做与集群无关的部分，也可以在已经
# 拿到 GPU 的作业里直接跑：
#     srun --gres=gpu:pro6000:1 --cpus-per-task=8 --time=02:00:00 \
#          bash scripts/eval/eval_e00_qwen3.sh
#
# 环境变量（都有默认值，集群侧由 eval.sbatch 覆写）：
#   MIMO_API_KEY  必填。用户模拟器密钥，只走环境变量，不写进任何文件
#   POLICY_MODEL  默认官方 Qwen3-8B
#   RUN_DIR       默认 experiments/e00_qwen3_baseline/run-$SLURM_JOB_ID
#   RAY_TMPDIR    默认 $RUN_DIR/ray
#   EXP_NAME      默认 E00_qwen3_base_$SLURM_JOB_ID
#
# 产物：RUN_DIR 下的 eval_trajectories.jsonl / tool_audit.jsonl / api.jsonl /
# metrics.jsonl / eval.log；收尾自动生成 baseline_summary.json 与
# baseline_report.md，其中 accepted=true 才算通过完整验收。
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/../.."
export REPO=$(pwd)

# 安全闸门：GPU 负载不能落在登录节点；密钥不能出现在命令行
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "拒绝执行：GPU 负载必须走 Slurm。" >&2
    echo "  sbatch cluster_setup/e00_qwen3_baseline/eval.sbatch" >&2
    echo "  srun --gres=gpu:pro6000:1 --cpus-per-task=8 --time=02:00:00 bash scripts/eval/eval_e00_qwen3.sh" >&2
    exit 2
fi
test -n "${MIMO_API_KEY:-}" || { echo "缺少 MIMO_API_KEY（不要写进命令行/配置文件）" >&2; exit 2; }

E00_DIR=$REPO/experiments/e00_qwen3_baseline
SPLIT_FILE=$E00_DIR/split.json
CONFIG_DIR=$REPO/configs/eval/e00

export POLICY_MODEL=${POLICY_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B}
export RUN_DIR=${RUN_DIR:-$E00_DIR/run-$SLURM_JOB_ID}
export RAY_TMPDIR=${RAY_TMPDIR:-$RUN_DIR/ray}
export EXP_NAME=${EXP_NAME:-E00_qwen3_base_$SLURM_JOB_ID}
# 用户模拟器进度日志的分母，见 configs/interaction_config/tau_bench_airline_mimo.yaml
export EXPECTED_TRAJECTORIES=320

umask 077
mkdir -p "$RUN_DIR"/{cache/triton,cache/torchinductor,cache/vllm,cache/cuda} "$RAY_TMPDIR"

# 缓存全部落在 RUN_DIR，不污染仓库与共享目录
export CUDA_CACHE_PATH=$RUN_DIR/cache/cuda
export TRITON_CACHE_DIR=$RUN_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$RUN_DIR/cache/torchinductor
export VLLM_CACHE_ROOT=$RUN_DIR/cache/vllm

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENAI_API_KEY=local-unused
export DS_SKIP_TRITON=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1 WANDB_MODE=disabled SWANLAB_MODE=disabled
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# 运行期埋点：summarize_e00.py 靠这几个文件出报告
export MIMO_METRICS_PATH=$RUN_DIR/api.jsonl
export MIMO_TRAJECTORY_PATH=$RUN_DIR/trajectories.jsonl
export TOOL_AUDIT_PATH=$RUN_DIR/tool_audit.jsonl
export E00_TRAJECTORIES_PATH=$RUN_DIR/eval_trajectories.jsonl
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl

# --- 收尾：写退出码 + 出报告 ------------------------------------------------
finish() {
    rc=$?
    trap - EXIT
    date +%s > "$RUN_DIR/job_finished.epoch"
    echo "$rc" > "$RUN_DIR/exit_code.txt"
    cp -f "$SPLIT_FILE" "$RUN_DIR/split.json"
    python "$REPO/scripts/eval/summarize_e00.py" "$RUN_DIR" || rc=1
    echo "E00_JOB_EXIT=$rc  dir=$RUN_DIR"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
date +%s > "$RUN_DIR/job_started.epoch"

# --- GPU 自检 ---------------------------------------------------------------
python -c 'import torch; assert torch.cuda.device_count() == 1; print("GPU:", torch.cuda.get_device_name(0), flush=True)'

# --- 生成 40 条评测输入（复用仓库既有的 build_rows + 冻结的 split.json）-----
python - "$RUN_DIR" "$SPLIT_FILE" <<'PY'
import json, runpy, sys
from pathlib import Path
import pandas as pd

run_dir, split_path = Path(sys.argv[1]), Path(sys.argv[2])
build_rows = runpy.run_path("scripts/train/grpo/build_grpo_parquet.py")["build_rows"]
split = json.loads(split_path.read_text(encoding="utf-8"))

rows = build_rows(split["train_task_ids"], "train") + build_rows(split["dev_task_ids"], "dev")
assert len(rows) == 40, len(rows)
assert not {r["extra_info"]["task_id"] for r in rows} & set(split["test_task_ids"])
pd.DataFrame(rows).to_parquet(run_dir / "eval.parquet", index=False)

# val_only 不会训练，但 _create_dataloader 一定要能读到 train_files，所以补一个占位文件
pd.DataFrame(build_rows(split["train_task_ids"], "train")).to_parquet(
    run_dir / "train-unused.parquet", index=False
)
print(f"PARQUET_READY train={len(split['train_task_ids'])} dev={len(split['dev_task_ids'])} test=0", flush=True)
PY

# --- 开跑（--config-path 指向 configs/eval/e00，它同时含 eval_qwen3.yaml / agent_loop.yaml）
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name=eval_qwen3 \
    2>&1 | tee "$RUN_DIR/eval.log"

echo "E00_MAIN_DONE 查看进度: grep 'MIMO rollout progress' $RUN_DIR/rank-0.out"
