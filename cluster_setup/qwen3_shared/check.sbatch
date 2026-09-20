#!/bin/bash -l
#SBATCH --job-name=qwen3-cpu-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cpu_ok
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -eo pipefail
module purge
source /projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/setup/activate.sh
: "${SOURCE_ROOT:?Set immutable source snapshot}" "${RUN_ROOT:?Set output directory}"
export REPO=$SOURCE_ROOT/agentic-grpo-longhorizon
export PYTHONPATH="$SOURCE_ROOT/verl:$SOURCE_ROOT/tau-bench:$REPO:$AGENTRL_ROOT/runs/grpo_probe_20260919/triton-3.3.1${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
srun bash "$REPO/${TASK_ENTRY:-scripts/train/grpo/check_qwen3.sh}"
