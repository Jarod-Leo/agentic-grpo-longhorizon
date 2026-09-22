#!/bin/bash
# CPU-only stage validation. DISTILLATION_CONFIGS selects already implemented stages.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export DISTILLATION_CONFIGS=${DISTILLATION_CONFIGS:-qwen3_opd}
source scripts/train/grpo/check_qwen3.sh
python -m pytest -q --junitxml="$RUN_ROOT/distillation-tests.xml" src/models/tests/test_token_teacher.py src/training/tests/test_distillation.py \
    ../verl/tests/trainer/ppo/test_distillation_loss_on_cpu.py
python - "$RUN_ROOT" <<'PY_CHECK'
import copy
import json
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.models.token_teacher import tokenizer_fingerprint
from verl.utils.config import omega_conf_to_dataclass

root = Path(os.environ["RUN_ROOT"])
names = os.environ.get("DISTILLATION_CONFIGS", "qwen3_opd").split(",")
with initialize_config_dir(config_dir=str(Path("configs/train/grpo").resolve()), version_base=None):
    baseline = OmegaConf.to_container(compose(config_name="qwen3_formal"), resolve=True)
    for name in names:
        config = compose(config_name=name)
        resolved = OmegaConf.to_container(config, resolve=True)
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        assert actor.distillation.enabled
        assert not resolved["actor_rollout_ref"]["actor"]["use_kl_loss"]
        assert resolved["actor_rollout_ref"]["actor"]["entropy_coeff"] == 0
        comparable = copy.deepcopy(resolved)
        comparable.pop("distillation")
        comparable["actor_rollout_ref"]["actor"].pop("distillation")
        assert comparable == baseline, name + " changed baseline outside distillation"
        (root / (name + "_resolved.json")).write_text(json.dumps(resolved, indent=2) + "\n")
        print("DISTILLATION_CONFIG_OK", name, "rl_coef", actor.distillation.rl_coef, "coef", actor.distillation.coef)
student = AutoTokenizer.from_pretrained(os.environ["POLICY_MODEL"], local_files_only=True)
teacher_path = os.environ.get("TEACHER_MODEL", "/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-32B")
teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
assert tokenizer_fingerprint(student) == tokenizer_fingerprint(teacher), "Actual Qwen3 tokenizers/templates differ"
print("ACTUAL_QWEN3_TOKENIZERS_COMPATIBLE", tokenizer_fingerprint(student))
PY_CHECK
for config in ${DISTILLATION_CONFIGS//,/ }; do
    mkdir -p "$RUN_ROOT/$config-input"
    python scripts/train/grpo/prepare_qwen3_run.py "$RUN_ROOT/$config-input" --mode train --config-name "$config" \
        --steps 200 --save-freq 50 --eval-freq 100 --eval-samples 8
 done
