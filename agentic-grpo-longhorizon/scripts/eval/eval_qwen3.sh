#!/bin/bash
# Shared evaluation for the base model and ADAPTER_PATH checkpoints.
set -eo pipefail
# Budget-limited training can finish at a variable step. Resolve only after
# the training job has completed, using its existing checkpoint tracker.
if [ -n "${EVAL_CHECKPOINT_ROOT:-}" ]; then
    test -z "${ADAPTER_PATH:-}" || { echo 'Set ADAPTER_PATH or EVAL_CHECKPOINT_ROOT, not both' >&2; exit 2; }
    checkpoint_step=$(cat "$EVAL_CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")
    [[ "$checkpoint_step" =~ ^[0-9]+$ ]] || exit 2
    export ADAPTER_PATH="$EVAL_CHECKPOINT_ROOT/global_step_$checkpoint_step/actor/lora_adapter"
fi
source "$(dirname "$0")/../train/grpo/qwen3_runtime.sh"
setup_qwen3_run eval "$@"
python -m verl.trainer.main_ppo \
    --config-path="$REPO/configs/eval/qwen3" --config-name=eval_qwen3 \
    "${QWEN3_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
