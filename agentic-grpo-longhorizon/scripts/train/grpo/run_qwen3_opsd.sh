#!/bin/bash
set -eo pipefail
export QWEN3_CONFIG=qwen3_opsd
exec bash "$(dirname "$0")/run_qwen3_formal.sh" "$@"
