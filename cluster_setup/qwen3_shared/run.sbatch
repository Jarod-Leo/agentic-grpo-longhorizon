#!/bin/bash -l
#SBATCH --job-name=qwen3-mimo
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --constraint=highmem
#SBATCH --exclude=gpu-pro6000-2
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60
set -eo pipefail
umask 077
module purge
source /projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/setup/activate.sh
: "${SOURCE_ROOT:?Set immutable source snapshot}" "${RUN_ROOT:?Set output directory}" "${TASK_ENTRY:?Set main-project script}"
export REPO=$SOURCE_ROOT/agentic-grpo-longhorizon
export PYTHONPATH="$SOURCE_ROOT/verl:$SOURCE_ROOT/tau-bench:$REPO:$AGENTRL_ROOT/runs/grpo_probe_20260919/triton-3.3.1${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    --output="$RUN_ROOT/rank-0.out" --error="$RUN_ROOT/rank-0.err" \
    bash "$REPO/$TASK_ENTRY"
