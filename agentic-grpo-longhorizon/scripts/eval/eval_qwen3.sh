#!/bin/bash
# Shared evaluation for the base model and ADAPTER_PATH checkpoints.
set -eo pipefail
source "$(dirname "$0")/../train/grpo/qwen3_runtime.sh"
setup_qwen3_run eval
python -m verl.trainer.main_ppo \
    --config-path="$REPO/configs/eval/qwen3" --config-name=eval_qwen3 \
    "${QWEN3_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
