#!/bin/bash
set -eo pipefail
export QWEN3_CONFIG=qwen3_grpo_opd
exec bash "$(dirname "$0")/run_qwen3_formal.sh" "$@"
