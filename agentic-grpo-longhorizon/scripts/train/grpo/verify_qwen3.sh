#!/bin/bash
# Three continuous steps, independent resume from step 2, then fixed step-3 evaluation.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export RUN_ROOT=${RUN_ROOT:-$(pwd)/experiments/e01_vanilla_grpo/run-$SLURM_JOB_ID}
export STEPS=3 SAMPLES=8 EVAL_SPLIT=train
unset RESUME_FROM ADAPTER_PATH
RUN_DIR="$RUN_ROOT/continuous" bash scripts/train/grpo/run_qwen3.sh train
RUN_DIR="$RUN_ROOT/resume" RESUME_FROM="$RUN_ROOT/continuous/checkpoints/global_step_2" \
    bash scripts/train/grpo/run_qwen3.sh train
python scripts/train/grpo/check_qwen3_readiness.py "$RUN_ROOT"
RUN_DIR="$RUN_ROOT/eval_step3" ADAPTER_PATH="$RUN_ROOT/continuous/checkpoints/global_step_3/actor/lora_adapter" \
    bash scripts/eval/eval_qwen3.sh
