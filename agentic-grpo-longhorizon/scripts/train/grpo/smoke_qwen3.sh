#!/bin/bash
# One real update, checkpoint archival, and adapter verification for any Qwen3 method.
set -eo pipefail
cd "$(dirname "$0")/../../.."
: "${RUN_ROOT:?Set smoke output directory}" "${CHECKPOINT_ARCHIVE_ROOT:?Set HDD archive directory}"
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/train}
export QWEN3_CONFIG=${QWEN3_CONFIG:-qwen3_formal}
export STEPS=1 SAMPLES=8 SAVE_FREQ=1 EVAL_FREQ=-1 EVAL_SPLIT=train
unset RESUME_FROM ADAPTER_PATH
bash scripts/train/grpo/run_qwen3.sh "$@"
checks=(--single-run)
if [ -n "${EXPECTED_LORA_INIT_SHA256:-}" ]; then
    checks+=(--expected-lora-sha256 "$EXPECTED_LORA_INIT_SHA256")
fi
python scripts/train/grpo/check_qwen3_readiness.py "$RUN_DIR" "${checks[@]}"
if [ "${VERIFY_RESUME:-0}" = 1 ]; then
    smoke_train_dir=$RUN_DIR
    smoke_archive_root=$CHECKPOINT_ARCHIVE_ROOT
    export RESUME_FROM=${CHECKPOINT_ROOT:-$smoke_train_dir/checkpoints}/global_step_1
    export RUN_DIR=$RUN_ROOT/resume
    export CHECKPOINT_ROOT=$RUN_DIR/checkpoints
    export CHECKPOINT_ARCHIVE_ROOT=$smoke_archive_root/resume
    export STEPS=2
    bash scripts/train/grpo/run_qwen3.sh "$@"
    python scripts/train/grpo/check_qwen3_readiness.py "$smoke_train_dir" --resume-run "$RUN_DIR"
fi
