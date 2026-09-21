#!/bin/bash
# Full checkpoints go to the explicitly configured HDD archive.
set -eo pipefail
: "${CHECKPOINT_ARCHIVE_ROOT:?Set the HDD checkpoint archive directory}"
export STEPS=200 SAMPLES=8 EVAL_SPLIT=train SAVE_FREQ=50 EVAL_FREQ=100 EVAL_SAMPLES=8
export QWEN3_CONFIG=qwen3_formal
exec bash "$(dirname "$0")/run_qwen3.sh"
