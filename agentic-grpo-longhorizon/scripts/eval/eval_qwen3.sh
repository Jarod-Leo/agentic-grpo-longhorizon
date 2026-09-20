#!/bin/bash
set -eo pipefail
exec bash "$(dirname "$0")/../train/grpo/run_qwen3.sh" eval "$@"
