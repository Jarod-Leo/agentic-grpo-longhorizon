#!/bin/bash
# CPU-only acceptance checks for experiment C.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
export RUN_DIR=${RUN_DIR:-/tmp/local-branch-training-check}
export EXP_NAME=local_branch_training_cpu
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/local-branch-ray}
export POLICY_MODEL=${POLICY_MODEL:-unused-for-config-check}
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

bash -n scripts/train/grpo/run_verified_local_credit.sh
python -m pytest -q \
    src/envs/tests/test_tau_bench_snapshot.py \
    src/evaluation/tests/test_local_branching.py \
    src/evaluation/tests/test_local_branch_training.py \
    ../verl/tests/trainer/ppo/test_local_branch_training_on_cpu.py \
    ../verl/tests/workers/actor/test_local_branch_loss_on_cpu.py

python - <<'PY'
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass

root = Path(os.environ["REPO"])
with initialize_config_dir(
    config_dir=str(root / "configs/train/grpo"), version_base=None
):
    config = compose(config_name="qwen3_verified_sft_local")
actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
assert actor.local_branch_loss.enabled
assert actor.local_branch_loss.coef == 0.25
assert actor.use_dynamic_bsz
assert actor.loss_agg_mode == "seq-mean-token-mean"
assert actor.ppo_epochs == 1
assert rollout.n == 8
assert rollout.multi_turn.local_branch.trigger == "critical_prefix"
assert rollout.multi_turn.local_branch.candidates == 4
assert rollout.multi_turn.local_branch.continuations == 2
assert rollout.multi_turn.local_branch.emit_training_fields
print("LOCAL_BRANCH_TRAINING_CONFIG_OK")
PY

echo LOCAL_BRANCH_TRAINING_CPU_OK
