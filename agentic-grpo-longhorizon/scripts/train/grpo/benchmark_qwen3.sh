#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../../.."
python scripts/train/grpo/benchmark_qwen3.py
