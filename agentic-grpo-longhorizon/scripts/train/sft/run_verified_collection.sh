#!/bin/bash
# Collect a fixed teacher or student-correction budget through the shared tau-bench loop.
set -eo pipefail

MODE=${1:?Usage: run_verified_collection.sh teacher|correction [Hydra overrides...]}
shift
case "$MODE" in
    teacher|correction) ;;
    *) echo "Unknown collection mode: $MODE" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export REPO=$PROJECT_ROOT
source "$SCRIPT_DIR/../grpo/qwen3_runtime.sh"

SMOKE_TASK_IDS=0,1,4,6,13,45
if [ "${COLLECTION_SCOPE:-smoke}" = formal ]; then
    COLLECTION_TASK_IDS=${COLLECTION_TASK_IDS:-$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["seen_task_ids"])))' "$PROJECT_ROOT/experiments/sft_collect_airline/split.json")}
    if [ "$MODE" = teacher ]; then
        COLLECTION_SAMPLES=${COLLECTION_SAMPLES:-4}
    else
        COLLECTION_SAMPLES=${COLLECTION_SAMPLES:-2}
    fi
else
    COLLECTION_TASK_IDS=${COLLECTION_TASK_IDS:-$SMOKE_TASK_IDS}
    COLLECTION_SAMPLES=${COLLECTION_SAMPLES:-2}
fi
export RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/experiments/e02_verified_sft/run-${SLURM_JOB_ID:-manual}}
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/collect-$MODE-${COLLECTION_SCOPE:-smoke}}

if [ "$MODE" = correction ]; then
    export ADAPTER_PATH=${ADAPTER_PATH:-${STUDENT_ADAPTER_PATH:?Set STUDENT_ADAPTER_PATH for correction collection}}
fi
export SAMPLES=$COLLECTION_SAMPLES EVAL_SPLIT=train TAU_REWARD_MODE=binary

COLLECTION_OVERRIDES=(
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO/configs/interaction_config/tau_bench_verified_sft_loop.yaml"
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REPO/configs/interaction_config/tau_bench_airline_mimo_v26.yaml"
    "++verified_sft_collection.mode=$MODE"
    '++verified_sft_collection.teacher_model=mimo-v2.6-pro'
    '++verified_sft_collection.teacher_max_tokens=1024'
    '++verified_sft_collection.rpm=100'
    '++verified_sft_collection.tpm=10000000'
    '++verified_sft_collection.max_inflight=32'
    '++verified_sft_collection.limiter_group=verified_sft_mimo_v26'
)
setup_qwen3_run eval "${COLLECTION_OVERRIDES[@]}" "$@"
python "$REPO/scripts/train/sft/prepare_verified_collection.py" "$RUN_DIR" \
    --mode "$MODE" --task-ids "$COLLECTION_TASK_IDS" --samples "$COLLECTION_SAMPLES"
export EXPECTED_TRAJECTORIES=$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RUN_DIR"],"run.json")))["expected_trajectories"])')
export VERIFIED_SFT_FAILURES_PATH=$RUN_DIR/collection_failures.jsonl

python -m verl.trainer.main_ppo \
    --config-path="$REPO/configs/eval/qwen3" --config-name=eval_qwen3 \
    "${QWEN3_OVERRIDES[@]}" "${COLLECTION_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"

python "$REPO/scripts/train/sft/build_verified_sft.py" \
    --input "$TRAJECTORIES_PATH" \
    --split "$REPO/experiments/sft_collect_airline/split.json" \
    --mode "$MODE" \
    --output "$RUN_DIR/verified_${MODE}.parquet" \
    --summary "$RUN_DIR/verified_${MODE}.summary.json"
