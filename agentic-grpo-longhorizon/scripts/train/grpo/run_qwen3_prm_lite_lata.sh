#!/bin/bash
# E05 selects the joint method; the shared formal entry owns the 200-step schedule.
set -eo pipefail
export QWEN3_CONFIG=qwen3_prm_lite_lata
exec bash "$(dirname "$0")/run_qwen3_formal.sh"
