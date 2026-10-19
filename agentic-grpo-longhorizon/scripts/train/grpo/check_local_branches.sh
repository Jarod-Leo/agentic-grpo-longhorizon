#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export RUN_DIR=$RUN_ROOT/config-check EXP_NAME=local_branch_cpu RAY_TMPDIR=/tmp/local-branch-config
mkdir -p "$RUN_DIR"
python -m pytest -q \
    src/envs/tests/test_tau_bench_snapshot.py \
    src/evaluation/tests/test_local_branch*.py \
    src/envs/tests/test_agent_loop_lifecycle.py \
    src/envs/tests/test_tau_bench_joint_loop.py \
    src/envs/tests/test_tau_bench_input.py
python - <<'PY'
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass

root = Path(os.environ['REPO'])
with initialize_config_dir(config_dir=str(root / 'configs/eval/qwen3'), version_base=None):
    config = compose(config_name='eval_qwen3', overrides=[
        '++actor_rollout_ref.rollout.multi_turn.local_branch.enable=true',
        '++actor_rollout_ref.rollout.multi_turn.local_branch.task_ids=[0,1,2,4]',
        '++actor_rollout_ref.rollout.multi_turn.local_branch.anchor_turns=[2,4,6]',
    ])
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    assert rollout.multi_turn.local_branch.enable
    assert list(rollout.multi_turn.local_branch.anchor_turns) == [2,4,6]
    assert config.trainer.val_only
    assert not config.data.apply_chat_template_kwargs.enable_thinking
    OmegaConf.save(config, Path(os.environ['RUN_DIR']) / 'config.yaml')
print('LOCAL_BRANCH_CONFIG_OK')
PY
tooling=/projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/repo/agentic-grpo-longhorizon/experiments/grpo_dynamic_sampling/tooling
cd "$REPO/../verl"
PYTHONPATH="$tooling:$PYTHONPATH" python -m ruff check verl/workers/config/rollout.py
PYTHONPATH="$tooling:$PYTHONPATH" python -m ruff format --check verl/workers/config/rollout.py
echo LOCAL_BRANCH_CPU_OK
