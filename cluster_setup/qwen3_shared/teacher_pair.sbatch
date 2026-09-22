#!/bin/bash -l
#SBATCH --job-name=qwen3-distillation
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --ntasks=2
#SBATCH --time=36:00:00
#SBATCH --signal=B:TERM@60
set -eo pipefail
umask 077
module purge
source /projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/setup/activate.sh
: "${SOURCE_ROOT:?Set immutable source snapshot}" "${RUN_ROOT:?Set output directory}"
export REPO=$SOURCE_ROOT/agentic-grpo-longhorizon
export PYTHONPATH="$SOURCE_ROOT/verl:$SOURCE_ROOT/tau-bench:$REPO:$AGENTRL_ROOT/runs/grpo_probe_20260919/triton-3.3.1${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    --output="$RUN_ROOT/rank-%t.out" --error="$RUN_ROOT/rank-%t.err" \
    bash "$REPO/scripts/train/grpo/run_qwen3_teacher_pair.sh"
