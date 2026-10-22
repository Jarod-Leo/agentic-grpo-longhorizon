#!/bin/bash
# CPU checks for the verified-SFT/local-credit additions and shared baseline.
set -eo pipefail
cd "$(dirname "$0")/../../.."
: "${SLURM_JOB_ID:?Run checks inside Slurm}" "${RUN_ROOT:?Set check output directory}"
export REPO=$(pwd)
export PYTHONPATH="$REPO/../verl:$REPO/../tau-bench:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
bash scripts/train/grpo/check_qwen3.sh
python -m pytest -q \
    src/envs/tests/test_verified_sft_collection.py \
    src/training/tests/test_verified_sft_dataset.py \
    src/training/tests/test_build_verified_sft.py \
    src/envs/tests/test_tau_bench_terminal.py \
    src/envs/tests/test_tau_bench_snapshot.py \
    src/evaluation/tests/test_local_branching.py \
    src/evaluation/tests/test_local_branch_metrics.py \
    src/evaluation/tests/test_local_branch_training.py \
    ../verl/tests/trainer/ppo/test_local_branch_training_on_cpu.py \
    ../verl/tests/workers/actor/test_local_branch_loss_on_cpu.py

export RUN_DIR="$RUN_ROOT/comparison-config" EXP_NAME=comparison_config
export RAY_TMPDIR="/tmp/verified-comparison-$SLURM_JOB_ID"
export CHECKPOINT_ROOT="$RUN_ROOT/checkpoints" CHECKPOINT_ARCHIVE_ROOT=
export VERIFIED_SFT_DATA="$RUN_ROOT/verified.parquet" VERIFIED_SFT_OUTPUT="$RUN_ROOT/sft"
python - <<'PY'
from pathlib import Path
import os
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer
from src.envs.verified_sft_agent_loop import render_teacher_response_tokens

with initialize_config_dir(config_dir=str(Path("configs/train/grpo").resolve()), version_base=None):
    baseline = compose(config_name="qwen3_formal")
    local = compose(config_name="qwen3_verified_sft_local")
    assert local.actor_rollout_ref.actor.local_branch_loss.enabled
    assert local.actor_rollout_ref.actor.local_branch_loss.coef == 0.25
    for key in ("optim", "ppo_epochs", "use_dynamic_bsz", "loss_agg_mode", "ppo_mini_batch_size"):
        assert baseline.actor_rollout_ref.actor[key] == local.actor_rollout_ref.actor[key], key
    assert local.actor_rollout_ref.rollout.multi_turn.local_branch.emit_training_fields
    assert local.actor_rollout_ref.rollout.multi_turn.local_branch.trigger == "critical_prefix"
with initialize_config_dir(config_dir=str(Path("configs/train/sft").resolve()), version_base=None):
    sft = compose(config_name="qwen3_verified_sft")
    assert sft.data.custom_cls.name == "VerifiedTokenSFTDataset"
    assert sft.model.lora_rank == 16 and sft.model.lora_alpha == 32
    assert sft.trainer.total_epochs == 2
    assert sft.trainer.resume_mode == "disable"
tokenizer = AutoTokenizer.from_pretrained(os.environ["POLICY_MODEL"], local_files_only=True)
for content in ("I can help with that.", '<tool_call>\n{"name":"get_user_details","arguments":{"user_id":"example"}}\n</tool_call>'):
    tokens = render_teacher_response_tokens(
        tokenizer, [{"role": "user", "content": "Please help with my reservation."}],
        [], content, {"enable_thinking": False},
    )
    assert tokenizer.decode(tokens, skip_special_tokens=True).strip() == content
    assert tokens[-1] == tokenizer.eos_token_id
print("VERIFIED_COMPARISON_CONFIG_OK")
PY

bash -n scripts/train/grpo/run_verified_comparison.sh \
    scripts/train/grpo/run_verified_local_credit.sh \
    scripts/train/sft/run_verified_collection.sh scripts/train/sft/run_verified_sft.sh

export PYTHONPATH="/projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/repo/agentic-grpo-longhorizon/experiments/grpo_dynamic_sampling/tooling:$PYTHONPATH"
cd "$REPO/../verl"
changed=(
    verl/experimental/agent_loop/tool_agent_loop.py
    verl/trainer/fsdp_sft_trainer.py
    verl/trainer/ppo/ray_trainer.py
    verl/workers/actor/dp_actor.py
    verl/workers/config/actor.py
    verl/workers/config/rollout.py
    tests/trainer/ppo/test_local_branch_training_on_cpu.py
    tests/workers/actor/test_local_branch_loss_on_cpu.py
)
python -m ruff check "${changed[@]}"
python -m ruff format --check "${changed[@]}"
if [ "${CHECK_TEACHER_API:-0}" = 1 ]; then
    cd "$REPO"
    export RUN_DIR="$RUN_ROOT/teacher-api"
    mkdir -p "$RUN_DIR"
    python scripts/train/sft/check_verified_sft.py --teacher-api
fi
echo VERIFIED_COMPARISON_CPU_CHECKS_PASSED
