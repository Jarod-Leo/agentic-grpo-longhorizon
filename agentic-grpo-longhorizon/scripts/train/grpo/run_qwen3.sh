#!/bin/bash
# Shared veRL runner: train [Hydra overrides...] or eval [Hydra overrides...].
set -eo pipefail
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
MODE=${1:-train}
shift || true
case "$MODE" in train|eval) ;; *) echo 'Expected train or eval' >&2; exit 2;; esac
test -n "${SLURM_JOB_ID:-}" || { echo 'Run GPU work inside Slurm' >&2; exit 2; }
test -n "${MIMO_API_KEY:-}" || { echo 'Activate setup/activate.sh to load MiMo credentials' >&2; exit 2; }
export RUN_DIR=${RUN_DIR:-$REPO/experiments/e01_vanilla_grpo/run-$SLURM_JOB_ID/$MODE}
export EXP_NAME=${EXP_NAME:-qwen3_${MODE}_${SLURM_JOB_ID}}
export RAY_TMPDIR=${RAY_TMPDIR:-/projects/_ssd/jiatian001ssd/q-$SLURM_JOB_ID}
export POLICY_MODEL=${POLICY_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B}
export PYTHONPATH="$REPO/../verl:$REPO/../tau-bench:$REPO${PYTHONPATH:+:$PYTHONPATH}"
umask 077
mkdir -p "$RUN_DIR" "$RAY_TMPDIR"
test ! -e "$RUN_DIR/run.json" || { echo "Refusing to overwrite $RUN_DIR" >&2; exit 2; }
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$RUN_DIR/cache/cuda}
export TRITON_CACHE_DIR=$RUN_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$RUN_DIR/cache/torchinductor VLLM_CACHE_ROOT=$RUN_DIR/cache/vllm
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENAI_API_KEY=local-unused DS_SKIP_TRITON=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1 WANDB_MODE=disabled SWANLAB_MODE=disabled
export MIMO_METRICS_PATH=$RUN_DIR/api.jsonl MIMO_TRAJECTORY_PATH=$RUN_DIR/trajectories.jsonl
export TOOL_AUDIT_PATH=$RUN_DIR/tool_audit.jsonl TRAJECTORIES_PATH=$RUN_DIR/eval_trajectories.jsonl
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
finish() {
    rc=$?
    trap - EXIT
    date +%s > "$RUN_DIR/job_finished.epoch"
    echo "$rc" > "$RUN_DIR/exit_code.txt"
    python "$REPO/scripts/eval/summarize_qwen3.py" "$RUN_DIR" || rc=1
    echo "QWEN3_EXIT=$rc dir=$RUN_DIR"
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
date +%s > "$RUN_DIR/job_started.epoch"
STEPS=${STEPS:-3}
SAMPLES=${SAMPLES:-8}
prepare=(--mode "$MODE" --split "${EVAL_SPLIT:-train}" --steps "$STEPS" --samples "$SAMPLES")
overrides=()
if [ -n "${RESUME_FROM:-}" ]; then
    prepare+=(--resume "$RESUME_FROM")
    overrides+=(trainer.resume_mode=resume_path "trainer.resume_from_path=$RESUME_FROM")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    test "$MODE" = eval
    prepare+=(--adapter "$ADAPTER_PATH")
    overrides+=(actor_rollout_ref.model.lora_rank=16 "actor_rollout_ref.model.lora_adapter_path=$ADAPTER_PATH")
fi
python scripts/train/grpo/prepare_qwen3_run.py "$RUN_DIR" "${prepare[@]}"
export EXPECTED_TRAJECTORIES=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"run.json")))["expected_trajectories"])')
python -c 'import torch; assert torch.cuda.device_count() == 1; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
if [ "$MODE" = eval ]; then
    config_path="$REPO/configs/eval/qwen3"
    config_name=eval_qwen3
    overrides+=("actor_rollout_ref.rollout.val_kwargs.n=$SAMPLES")
else
    config_path="$REPO/configs/train/grpo"
    config_name=qwen3_mimo
    overrides+=("trainer.total_training_steps=$STEPS" "actor_rollout_ref.rollout.n=$SAMPLES")
fi
python -m verl.trainer.main_ppo --config-path="$config_path" --config-name="$config_name" "${overrides[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
