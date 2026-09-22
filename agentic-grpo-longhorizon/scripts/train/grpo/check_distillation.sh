#!/bin/bash
# CPU-only stage validation. DISTILLATION_CONFIGS selects already implemented stages.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export DISTILLATION_CONFIGS=${DISTILLATION_CONFIGS:-qwen3_opd}
source scripts/train/grpo/check_qwen3.sh
python -m pytest -q --junitxml="$RUN_ROOT/distillation-tests.xml" src/models/tests/test_token_teacher.py src/training/tests/test_distillation.py src/training/tests/test_self_distillation.py src/envs/tests/test_opsd_feedback.py \
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
    resolved_configs = {}
    expected_coefficients = {"qwen3_opd": (0.0, 1.0), "qwen3_grpo_opd": (1.0, 0.3), "qwen3_opsd": (0.0, 1.0), "qwen3_grpo_opsd": (1.0, 0.3)}
    for name in names:
        config = compose(config_name=name)
        resolved = OmegaConf.to_container(config, resolve=True)
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        assert actor.distillation.enabled
        assert (actor.distillation.rl_coef, actor.distillation.coef) == expected_coefficients[name]
        if name in {"qwen3_opsd", "qwen3_grpo_opsd"}:
            assert resolved["distillation"]["teacher_mode"] == "self_feedback"
            assert resolved["distillation"]["feedback_mode"] == "F2"
            assert "endpoint" not in resolved["distillation"]
        resolved_configs[name] = resolved
        assert not resolved["actor_rollout_ref"]["actor"]["use_kl_loss"]
        assert resolved["actor_rollout_ref"]["actor"]["entropy_coeff"] == 0
        comparable = copy.deepcopy(resolved)
        comparable.pop("distillation")
        comparable["actor_rollout_ref"]["actor"].pop("distillation")
        assert comparable == baseline, name + " changed baseline outside distillation"
        (root / (name + "_resolved.json")).write_text(json.dumps(resolved, indent=2) + "\n")
        print("DISTILLATION_CONFIG_OK", name, "rl_coef", actor.distillation.rl_coef, "coef", actor.distillation.coef)
if {"qwen3_opd", "qwen3_grpo_opd"}.issubset(resolved_configs):
    comparable = copy.deepcopy(resolved_configs["qwen3_grpo_opd"])
    comparable["actor_rollout_ref"]["actor"]["distillation"].update(rl_coef=0.0, coef=1.0)
    assert comparable == resolved_configs["qwen3_opd"], "E07 differs from E06 outside objective coefficients"
if {"qwen3_opsd", "qwen3_grpo_opsd"}.issubset(resolved_configs):
    comparable = copy.deepcopy(resolved_configs["qwen3_grpo_opsd"])
    comparable["actor_rollout_ref"]["actor"]["distillation"].update(rl_coef=0.0, coef=1.0)
    assert comparable == resolved_configs["qwen3_opsd"], "E12 differs from E11 outside objective coefficients"
student = AutoTokenizer.from_pretrained(os.environ["POLICY_MODEL"], local_files_only=True)
teacher_path = os.environ.get("TEACHER_MODEL", "/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-32B")
teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
assert tokenizer_fingerprint(student) == tokenizer_fingerprint(teacher), "Actual Qwen3 tokenizers/templates differ"
print("ACTUAL_QWEN3_TOKENIZERS_COMPATIBLE", tokenizer_fingerprint(student))
self_config = next((c for c in resolved_configs.values() if c["distillation"]["teacher_mode"] == "self_feedback"), None)
if self_config is not None:
    import numpy as np
    import torch
    from src.envs.opsd_feedback import build_opsd_feedback
    from src.training.self_distillation import build_self_teacher_batch
    from verl.protocol import DataProto
    state = {
        "contaminated": False,
        "action_history": [{"tool": "get_reservation_details", "parameters": {"reservation_id": "invalid"}, "is_error": True}] * 2,
    }
    feedback = build_opsd_feedback(state, outcome_reward=0.0, termination="assistant_turn_limit")
    prompt = student.encode("Original airline policy and user request.", add_special_tokens=False)
    response = student.encode("I will check the reservation details.", add_special_tokens=False)
    ids = torch.tensor([prompt + response])
    batch = DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
                 "responses": torch.tensor([response]), "response_mask": torch.ones((1, len(response)), dtype=torch.long)},
        non_tensors={"opsd_feedback": np.array([feedback], dtype=object)},
        meta_info={"distillation_policy_version": 0},
    )
    teacher_batch, metrics = build_self_teacher_batch(batch, self_config["distillation"], student)
    assert torch.equal(teacher_batch.batch["responses"], batch.batch["responses"])
    assert 0 < metrics["distillation/feedback_tokens"] <= 512
    print("ACTUAL_QWEN3_SELF_FEEDBACK_OK", metrics)
PY_CHECK
for config in ${DISTILLATION_CONFIGS//,/ }; do
    mkdir -p "$RUN_ROOT/$config-input"
    python scripts/train/grpo/prepare_qwen3_run.py "$RUN_ROOT/$config-input" --mode train --config-name "$config" \
        --steps 200 --save-freq 50 --eval-freq 100 --eval-samples 8
 done
