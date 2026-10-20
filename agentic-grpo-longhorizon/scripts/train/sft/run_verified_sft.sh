#!/bin/bash
# Train the B initialization adapter from already gated token-exact trajectories.
set -eo pipefail

SCOPE=${1:-formal}
case "$SCOPE" in
    smoke|formal) ;;
    *) echo "Usage: run_verified_sft.sh smoke|formal [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true

cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
test -n "${SLURM_JOB_ID:-}" || { echo 'Run SFT inside Slurm' >&2; exit 2; }
export POLICY_MODEL=${POLICY_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B}
export RUN_ROOT=${RUN_ROOT:-$REPO/experiments/e02_verified_sft/run-$SLURM_JOB_ID}
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/sft}
export VERIFIED_SFT_OUTPUT=${VERIFIED_SFT_OUTPUT:-$RUN_DIR/checkpoints}
export PYTHONPATH="$REPO/../verl:$REPO/../tau-bench:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR" "$VERIFIED_SFT_OUTPUT"

if [ "$SCOPE" = smoke ]; then
    export SFT_EPOCHS=${SFT_EPOCHS:-1}
    MIN_SFT_SAMPLES=2
else
    export SFT_EPOCHS=${SFT_EPOCHS:-2}
    MIN_SFT_SAMPLES=4
fi

finish_verified_sft() {
    rc=$?
    trap - EXIT
    finished=$(date +%s)
    echo "$finished" > "$RUN_DIR/job_finished.epoch"
    echo "$rc" > "$RUN_DIR/exit_code.txt"
    rows=0
    if [ -n "${VERIFIED_SFT_DATA:-}" ] && [ -f "$VERIFIED_SFT_DATA" ]; then
        rows=$(python -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "$VERIFIED_SFT_DATA")
    fi
    step=0
    if [ -f "$VERIFIED_SFT_OUTPUT/latest_checkpointed_iteration.txt" ]; then
        step=$(cat "$VERIFIED_SFT_OUTPUT/latest_checkpointed_iteration.txt")
    fi
    adapter=""
    if [ -f "$RUN_DIR/final_adapter.txt" ]; then adapter=$(cat "$RUN_DIR/final_adapter.txt"); fi
    export SFT_FINISHED=$finished SFT_EXIT_CODE=$rc SFT_ROWS=$rows SFT_STEP=$step SFT_ADAPTER=$adapter SFT_SCOPE=$SCOPE
    python -c 'import json,os; started=int(open(os.path.join(os.environ["RUN_DIR"],"job_started.epoch")).read()); finished=int(os.environ["SFT_FINISHED"]); out={"scope":os.environ["SFT_SCOPE"],"samples":int(os.environ["SFT_ROWS"]),"epochs":int(os.environ.get("SFT_EPOCHS","0")),"steps":int(os.environ["SFT_STEP"]),"adapter":os.environ["SFT_ADAPTER"] or None,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"elapsed_seconds":finished-started,"gpu_hours":(finished-started)/3600,"exit_code":int(os.environ["SFT_EXIT_CODE"])}; json.dump(out,open(os.path.join(os.environ["RUN_DIR"],"summary.json"),"w"),indent=2); open(os.path.join(os.environ["RUN_DIR"],"summary.json"),"a").write("\n")'
    exit "$rc"
}
date +%s > "$RUN_DIR/job_started.epoch"
trap finish_verified_sft EXIT

if [ -z "${VERIFIED_SFT_DATA:-}" ]; then
    : "${SFT_FILES:?Set SFT_FILES='[teacher.parquet,correction.parquet]'}"
    mapfile -t INPUT_FILES < <(python -c 'import os,yaml; value=yaml.safe_load(os.environ["SFT_FILES"]); assert isinstance(value,list) and len(value)==2; print(*value,sep="\n")')
    python "$REPO/scripts/train/sft/combine_verified_sft.py" \
        --teacher "${INPUT_FILES[0]}" --correction "${INPUT_FILES[1]}" \
        --output "$RUN_DIR/verified_sft.parquet" --min-samples "$MIN_SFT_SAMPLES"
    export VERIFIED_SFT_DATA=$RUN_DIR/verified_sft.parquet
fi
test -f "$VERIFIED_SFT_DATA"

python "$REPO/scripts/train/sft/check_verified_sft.py" \
    --data "$VERIFIED_SFT_DATA" --model "$POLICY_MODEL" --min-samples "$MIN_SFT_SAMPLES"

SFT_OVERRIDES=()
if [ "$SCOPE" = smoke ]; then
    SFT_OVERRIDES+=(data.train_batch_size=1 data.val_max_samples=1)
fi
SFT_OVERRIDES+=("trainer.total_epochs=$SFT_EPOCHS")
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path="$REPO/configs/train/sft" --config-name=qwen3_verified_sft \
    "${SFT_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/sft.log"

STEP=$(cat "$VERIFIED_SFT_OUTPUT/latest_checkpointed_iteration.txt")
ADAPTER="$VERIFIED_SFT_OUTPUT/global_step_$STEP/lora_adapter"
python "$REPO/scripts/train/sft/check_verified_sft.py" --adapter "$ADAPTER"
printf '%s\n' "$ADAPTER" > "$RUN_DIR/final_adapter.txt"
echo "VERIFIED_SFT_TRAINED scope=$SCOPE adapter=$ADAPTER"
