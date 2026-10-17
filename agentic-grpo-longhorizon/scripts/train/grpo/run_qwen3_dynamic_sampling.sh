#!/bin/bash
# Trajectory budget is configured in qwen3_dynamic_sampling.yaml.
set -eo pipefail
: "${CHECKPOINT_ARCHIVE_ROOT:?Set the HDD checkpoint archive directory}"
export STEPS=${STEPS:-200} SAMPLES=${SAMPLES:-8} EVAL_SPLIT=train
export SAVE_FREQ=${SAVE_FREQ:-50} EVAL_FREQ=${EVAL_FREQ:-100} EVAL_SAMPLES=${EVAL_SAMPLES:-8}
export QWEN3_CONFIG=qwen3_dynamic_sampling
exec bash "$(dirname "$0")/run_qwen3.sh" "$@"
