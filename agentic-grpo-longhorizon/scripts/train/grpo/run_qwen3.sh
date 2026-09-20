#!/bin/bash
# Qwen3 vanilla GRPO. Extra arguments are forwarded to Hydra.
set -eo pipefail
# Keep the earlier train|eval invocation compatible.
if [ "${1:-}" = eval ]; then
    shift
    exec bash "$(dirname "$0")/../../eval/eval_qwen3.sh" "$@"
fi
[ "${1:-}" != train ] || shift

source "$(dirname "$0")/qwen3_runtime.sh"
setup_qwen3_run train
python -m verl.trainer.main_ppo \
    --config-path="$REPO/configs/train/grpo" --config-name=qwen3_mimo \
    "${QWEN3_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
