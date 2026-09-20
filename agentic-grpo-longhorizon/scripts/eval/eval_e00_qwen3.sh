#!/bin/bash
# Compatibility entry: new E00 runs use protocol v2 and the shared evaluator.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export RUN_DIR=${RUN_DIR:-$REPO/experiments/e00_qwen3_baseline/protocol-v2-${SLURM_JOB_ID:?Run inside Slurm}}
exec bash "$REPO/scripts/eval/eval_qwen3.sh" "$@"
