#!/bin/bash
# One real update, checkpoint archival, and adapter verification for any Qwen3 method.
set -eo pipefail
cd "$(dirname "$0")/../../.."
: "${RUN_ROOT:?Set smoke output directory}" "${CHECKPOINT_ARCHIVE_ROOT:?Set HDD archive directory}"
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/train}
export QWEN3_CONFIG=${QWEN3_CONFIG:-qwen3_formal}
export STEPS=1 SAMPLES=8 SAVE_FREQ=1 EVAL_FREQ=-1 EVAL_SPLIT=train
unset RESUME_FROM ADAPTER_PATH
bash scripts/train/grpo/run_qwen3.sh
checks=(--single-run)
if [ -n "${EXPECTED_LORA_INIT_SHA256:-}" ]; then
    checks+=(--expected-lora-sha256 "$EXPECTED_LORA_INIT_SHA256")
fi
python scripts/train/grpo/check_qwen3_readiness.py "$RUN_DIR" "${checks[@]}"
