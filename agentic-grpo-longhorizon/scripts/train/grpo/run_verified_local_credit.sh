#!/bin/bash
# Experiment C stages. Submit each stage in Slurm with dependencies.
set -eo pipefail
stage=${1:?Usage: run_verified_local_credit.sh smoke|signal|train}
shift
: "${RUN_ROOT:?Set the experiment-C output root}"
: "${INIT_ADAPTER_PATH:?Set the verified SFT adapter used by B and C}"

case "$stage" in
    smoke)
        export RUN_DIR=${RUN_DIR:-$RUN_ROOT/smoke}
        export STEPS=${STEPS:-1} SAMPLES=${SAMPLES:-8}
        export SAVE_FREQ=${SAVE_FREQ:-1} EVAL_FREQ=-1 EVAL_SPLIT=train
        export QWEN3_CONFIG=qwen3_verified_sft_local
        exec bash "$(dirname "$0")/run_qwen3.sh" "$@"
        ;;
    signal)
        export RUN_DIR=${RUN_DIR:-$RUN_ROOT/signal}
        export ADAPTER_PATH=$INIT_ADAPTER_PATH
        unset INIT_ADAPTER_PATH
        export SAMPLES=1 EVAL_SPLIT=train
        source "$(dirname "$0")/qwen3_runtime.sh"
        branch_overrides=(
            'actor_rollout_ref.rollout.agent.agent_loop_config_path=${oc.env:REPO}/configs/interaction_config/tau_bench_local_branch_loop.yaml'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.enable=true'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.validation_only=true'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.trigger=critical_prefix'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.task_ids=[0,1,4,6,13,45]'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.candidates=4'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.continuations=8'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.emit_training_fields=false'
            '++actor_rollout_ref.rollout.multi_turn.local_branch.output_dir=${oc.env:RUN_DIR}/branches'
            'actor_rollout_ref.rollout.val_kwargs.temperature=1.0'
            'actor_rollout_ref.rollout.val_kwargs.top_p=1.0'
        )
        setup_qwen3_run eval "${branch_overrides[@]}" "$@"
        python -m verl.trainer.main_ppo \
            --config-path="$REPO/configs/eval/qwen3" --config-name=eval_qwen3 \
            "${QWEN3_OVERRIDES[@]}" "${branch_overrides[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
        python "$REPO/scripts/eval/summarize_training_branches.py" \
            "$RUN_DIR" --candidates 4 --continuations 8
        ;;
    train)
        signal_summary=${LOCAL_SIGNAL_SUMMARY:-$RUN_ROOT/signal/local_branch_training_summary.json}
        stopped_path=$RUN_ROOT/stopped.json
        if ! python - "$signal_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
accepted = path.exists() and json.loads(path.read_text()).get("signal_gate_passed") is True
raise SystemExit(0 if accepted else 1)
PY
        then
            mkdir -p "$RUN_ROOT"
            printf '%s\n' '{"stage":"c_train","started":false,"reason":"local_signal_gate_failed"}' > "$stopped_path"
            echo "C_TRAIN_STOPPED reason=local_signal_gate_failed summary=$signal_summary"
            exit 0
        fi
        export RUN_DIR=${RUN_DIR:-$RUN_ROOT/train}
        export STEPS=${STEPS:-25} SAMPLES=${SAMPLES:-8}
        export SAVE_FREQ=${SAVE_FREQ:-25} EVAL_FREQ=${EVAL_FREQ:-25} EVAL_SPLIT=train
        export QWEN3_CONFIG=qwen3_verified_sft_local
        exec bash "$(dirname "$0")/run_qwen3.sh" "$@"
        ;;
    *)
        echo "Unknown C stage: $stage" >&2
        exit 2
        ;;
esac
