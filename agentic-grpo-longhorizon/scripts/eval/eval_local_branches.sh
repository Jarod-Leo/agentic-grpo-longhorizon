#!/bin/bash
# Frozen-policy branch diagnostics through the shared veRL evaluation runtime.
set -eo pipefail
export EVAL_SPLIT=train
export SAMPLES=${SAMPLES:-1}
export BRANCH_STAGE=${BRANCH_STAGE:-smoke}
export ADAPTER_PATH=${ADAPTER_PATH:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3/global_step_200/actor/lora_adapter}
if [ "$BRANCH_STAGE" = smoke ]; then
    export BRANCH_TASK_IDS=${BRANCH_TASK_IDS:-'[0,1,2,4]'}
else
    test "$BRANCH_STAGE" = signal
    export BRANCH_TASK_IDS=${BRANCH_TASK_IDS:-'[]'}
fi
export BRANCH_ANCHOR_TURNS=${BRANCH_ANCHOR_TURNS:-'[2]'}
source "$(dirname "$0")/../train/grpo/qwen3_runtime.sh"
branch_overrides=(
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO/configs/interaction_config/tau_bench_local_branch_loop.yaml"
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REPO/configs/interaction_config/tau_bench_airline_mimo_v26.yaml"
    'actor_rollout_ref.rollout.val_kwargs.temperature=1.0'
    'actor_rollout_ref.rollout.val_kwargs.top_p=1.0'
    '++actor_rollout_ref.rollout.multi_turn.local_branch.enable=true'
    '++actor_rollout_ref.rollout.multi_turn.local_branch.validation_only=true'
    '++actor_rollout_ref.rollout.multi_turn.local_branch.candidates=4'
    '++actor_rollout_ref.rollout.multi_turn.local_branch.continuations=2'
    "++actor_rollout_ref.rollout.multi_turn.local_branch.anchor_turns=$BRANCH_ANCHOR_TURNS"
    "++actor_rollout_ref.rollout.multi_turn.local_branch.task_ids=$BRANCH_TASK_IDS"
    '++actor_rollout_ref.rollout.multi_turn.local_branch.output_dir=${oc.env:RUN_DIR}/branches'
)
setup_qwen3_run eval "${branch_overrides[@]}" "$@"
# Branch points may be skipped; the actual terminal episode count is not fixed.
export EXPECTED_TRAJECTORIES=0

# Branches add audit/API records but are not ordinary independent eval episodes.
# Keep all costs and validate them with the branch-aware report.
finish_branch_run() {
    rc=$?
    trap - EXIT
    date +%s > "$RUN_DIR/job_finished.epoch"
    echo "$rc" > "$RUN_DIR/exit_code.txt"
    python "$REPO/scripts/eval/summarize_local_branches.py" "$RUN_DIR" || rc=1
    echo "LOCAL_BRANCH_EXIT=$rc dir=$RUN_DIR"
    exit "$rc"
}
trap finish_branch_run EXIT
python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ['RUN_DIR'])
meta = json.loads((run / 'run.json').read_text())
tasks = json.loads(os.environ['BRANCH_TASK_IDS']) or meta['train_task_ids']
record = {
    'stage': os.environ['BRANCH_STAGE'],
    'expected_backbones': 40 * int(os.environ['SAMPLES']),
    'task_ids': tasks,
    'candidates': 4,
    'continuations': 2,
    'anchor_turns': json.loads(os.environ['BRANCH_ANCHOR_TURNS']),
    'adapter_path': os.environ['ADAPTER_PATH'],
    'simulator_model': 'mimo-v2.6',
    'optimizer_updates': 0,
}
(run / 'branch_run.json').write_text(json.dumps(record, indent=2) + '\n')
PY
python -m verl.trainer.main_ppo \
    --config-path="$REPO/configs/eval/qwen3" --config-name=eval_qwen3 \
    "${QWEN3_OVERRIDES[@]}" "${branch_overrides[@]}" "$@" 2>&1 | tee "$RUN_DIR/run.log"
