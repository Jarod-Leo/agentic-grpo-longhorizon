#!/bin/bash
# Staged A/B/C comparison, using the shared collection, veRL and evaluation entries.
set -eo pipefail

STAGE=${1:?Usage: run_verified_comparison.sh smoke|collection_smoke|sft|b|c}
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
: "${SLURM_JOB_ID:?Run inside Slurm}" "${COMPARISON_ROOT:?Set experiment output root}"
: "${STUDENT_ADAPTER_PATH:?Set the corrected GRPO adapter used for repair prefixes}"
export WANDB_MODE=${WANDB_MODE:-disabled}
export CHECKPOINT_ARCHIVE_ROOT=
COMMON_OVERRIDES=(
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REPO/configs/interaction_config/tau_bench_airline_mimo_v26.yaml"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=65536"
)

# Each child owns its traps and logs. Avoid carrying one stage's paths into another.
run_child() {
    local output=$1
    shift
    env -u RUN_DIR -u CHECKPOINT_ROOT -u RESUME_FROM -u ADAPTER_PATH \
        -u INIT_ADAPTER_PATH -u EVAL_CHECKPOINT_ROOT -u VERIFIED_SFT_DATA \
        -u VERIFIED_SFT_OUTPUT RUN_ROOT="$output" "$@"
}

collect_and_sft() {
    local scope=$1 output=$2
    local teacher_data="$output/teacher/collect-teacher-$scope/verified_teacher.parquet"
    if [ "$scope" = formal ] && [ -n "${VERIFIED_TEACHER_DATA:-}" ]; then
        teacher_data=$VERIFIED_TEACHER_DATA
        test -f "$teacher_data"
        echo "REUSING_VERIFIED_TEACHER_DATA path=$teacher_data"
    else
        run_child "$output/teacher" COLLECTION_SCOPE="$scope" \
            bash scripts/train/sft/run_verified_collection.sh teacher
    fi
    run_child "$output/correction" COLLECTION_SCOPE="$scope" \
        bash scripts/train/sft/run_verified_collection.sh correction
    run_child "$output" \
        SFT_FILES="[$teacher_data,$output/correction/collect-correction-$scope/verified_correction.parquet]" \
        bash scripts/train/sft/run_verified_sft.sh "$scope"
}

train_b() {
    local adapter=$1 output=$2 steps=$3 eval_frequency=$4
    run_child "$output" INIT_ADAPTER_PATH="$adapter" QWEN3_CONFIG=qwen3_formal \
        STEPS="$steps" SAVE_FREQ="$steps" EVAL_FREQ="$eval_frequency" \
        EXP_NAME="verified_sft_grpo_${steps}" \
        bash scripts/train/grpo/run_qwen3.sh "${COMMON_OVERRIDES[@]}"
}

evaluate() {
    local adapter=$1 output=$2 split=$3
    run_child "$output" ADAPTER_PATH="$adapter" EVAL_SPLIT="$split" SAMPLES=8 \
        EXP_NAME="verified_comparison_${STAGE}_${split}" \
        bash scripts/eval/eval_qwen3.sh "${COMMON_OVERRIDES[@]}"
}

case "$STAGE" in
    collection_smoke)
        # Revalidate the changed collection path; prior SFT/GRPO smoke remains valid.
        run_child "$COMPARISON_ROOT/smoke/correction" COLLECTION_SCOPE=smoke \
            bash scripts/train/sft/run_verified_collection.sh correction
        ;;
    smoke)
        collect_and_sft smoke "$COMPARISON_ROOT/smoke"
        adapter=$(cat "$COMPARISON_ROOT/smoke/sft/final_adapter.txt")
        train_b "$adapter" "$COMPARISON_ROOT/smoke/b" 1 -1
        ;;
    sft)
        collect_and_sft formal "$COMPARISON_ROOT"
        adapter=$(cat "$COMPARISON_ROOT/sft/final_adapter.txt")
        evaluate "$adapter" "$COMPARISON_ROOT/sft_eval/train" train
        evaluate "$adapter" "$COMPARISON_ROOT/sft_eval/test" test
        ;;
    b)
        adapter=$(cat "$COMPARISON_ROOT/sft/final_adapter.txt")
        train_b "$adapter" "$COMPARISON_ROOT/b" 25 25
        evaluate "$COMPARISON_ROOT/b/train/checkpoints/global_step_25/actor/lora_adapter" \
            "$COMPARISON_ROOT/b/test" test
        ;;
    c)
        adapter=$(cat "$COMPARISON_ROOT/sft/final_adapter.txt")
        for local_stage in smoke signal train; do
            if [ "$local_stage" = smoke ]; then
                # The smoke adapter is not reused; keep SSD space for formal checkpoints.
                run_child "$COMPARISON_ROOT/c" INIT_ADAPTER_PATH="$adapter" SAVE_FREQ=-1 \
                    bash scripts/train/grpo/run_verified_local_credit.sh "$local_stage" \
                    "${COMMON_OVERRIDES[@]}"
            else
                run_child "$COMPARISON_ROOT/c" INIT_ADAPTER_PATH="$adapter" \
                    bash scripts/train/grpo/run_verified_local_credit.sh "$local_stage" \
                    "${COMMON_OVERRIDES[@]}"
            fi
        done
        if [ ! -f "$COMPARISON_ROOT/c/stopped.json" ]; then
            evaluate "$COMPARISON_ROOT/c/train/checkpoints/global_step_25/actor/lora_adapter" \
                "$COMPARISON_ROOT/c/test" test
        fi
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac
if [ -f "$COMPARISON_ROOT/comparison_manifest.json" ]; then
    python scripts/eval/summarize_qwen3_suite.py "$COMPARISON_ROOT/comparison_manifest.json"
fi
echo "VERIFIED_COMPARISON_STAGE_COMPLETE stage=$STAGE job=$SLURM_JOB_ID"
