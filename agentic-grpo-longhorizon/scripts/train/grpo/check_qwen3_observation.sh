#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
export PYTHONPATH="$REPO/../verl:$REPO/../tau-bench:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export RUN_ROOT=${RUN_ROOT:-$REPO/experiments/e01_vanilla_grpo/observation-cpu-check}
export RUN_DIR=$RUN_ROOT/config-check EXP_NAME=observation_cpu_check
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/qwen3-observation-config}
export POLICY_MODEL=${POLICY_MODEL:-/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B}
export PREPARED_RUN=$RUN_ROOT/prepared
mkdir -p "$RUN_DIR"

python -m pytest -q \
    src/envs/tests/test_tool_observation.py \
    src/envs/tests/test_tau_bench_terminal.py \
    src/evaluation/tests/test_terminal_audit.py \
    src/evaluation/tests/test_qwen3_summary.py

mkdir -p "$PREPARED_RUN"
python scripts/train/grpo/prepare_qwen3_run.py "$PREPARED_RUN" \
    --mode train --config-name qwen3_formal --steps 1 \
    --override "actor_rollout_ref.rollout.multi_turn.interaction_config_path=$REPO/configs/interaction_config/tau_bench_airline_mimo_v26.yaml"

python - <<'PY'
import json
import os
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

root = Path(os.environ["REPO"])
interaction_path = root / "configs/interaction_config/tau_bench_airline_mimo_v26.yaml"
override = (
    "actor_rollout_ref.rollout.multi_turn.interaction_config_path="
    f"{interaction_path}"
)
with initialize_config_dir(
    config_dir=str(root / "configs/train/grpo"), version_base=None
):
    config = compose(config_name="qwen3_formal", overrides=[override])

assert config.tau_bench_reward_mode == "binary"
assert config.algorithm.adv_estimator == "grpo"
assert config.actor_rollout_ref.model.use_remove_padding is True
assert config.actor_rollout_ref.actor.use_dynamic_bsz is True
assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 32768
assert config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length == 65536
assert config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side == "middle"
assert Path(config.actor_rollout_ref.rollout.multi_turn.interaction_config_path) == interaction_path
interaction = yaml.safe_load(interaction_path.read_text())
assert interaction["interaction"][0]["config"]["user_model"] == "mimo-v2.6-flash"
metadata = json.loads((Path(os.environ["PREPARED_RUN"]) / "run.json").read_text())
assert metadata["max_tool_response_length"] == 65536
assert metadata["tool_response_truncate_side"] == "middle"
assert Path(metadata["interaction_config_path"]) == interaction_path

tool_path = Path(config.data.tool_config_path)
tools = [item["tool_schema"] for item in yaml.safe_load(tool_path.read_text())["tools"]]
tool_names = {item["function"]["name"] for item in tools}
assert {"get_reservation_details", "search_direct_flight", "book_reservation"} <= tool_names

import runpy

build = runpy.run_path(str(root / "scripts/train/grpo/build_grpo_parquet.py"))
messages = build["build_rows"]([0], "train")[0]["prompt"] + [
    {"role": "user", "content": "I need help with my reservation."}
]
tokenizer = AutoTokenizer.from_pretrained(
    config.actor_rollout_ref.model.path, local_files_only=True
)
tokens = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    enable_thinking=False,
    add_generation_prompt=True,
)
assert len(tokens) < config.data.max_prompt_length
assert tokenizer.decode(tokens).endswith("<think>\n\n</think>\n\n")
print(
    "QWEN3_OBSERVATION_CONFIG_OK",
    f"prompt_tokens={len(tokens)}",
    f"max_tool_response_length={config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length}",
)
PY
