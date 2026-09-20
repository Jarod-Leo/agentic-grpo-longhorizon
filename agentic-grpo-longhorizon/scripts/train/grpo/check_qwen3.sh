#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/../../.."
export REPO=$(pwd)
export PYTHONPATH="$REPO/../verl:$REPO/../tau-bench:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export RUN_DIR=$RUN_ROOT/config-check EXP_NAME=config_check RAY_TMPDIR=/tmp/qwen3-config-check
mkdir -p "$RUN_DIR"
python -m pytest -q src/envs/tests/test_tau_bench_input.py src/evaluation/tests/test_pass_metrics.py \
    src/evaluation/tests/test_qwen3_summary.py src/envs/tests/test_agent_loop_lifecycle.py src/evaluation/tests/test_e00_summary.py \
    ../verl/tests/trainer/ppo/test_grpo_signal_on_cpu.py
python scripts/train/grpo/prepare_qwen3_run.py "$RUN_DIR" --mode train
python - "$RUN_ROOT" <<'PY'
import sys,yaml
from pathlib import Path
from transformers import AutoTokenizer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
root=Path(sys.argv[1])
def config(directory, name, output):
    with initialize_config_dir(config_dir=str(Path(directory).resolve()), version_base=None):
        resolved=OmegaConf.to_container(compose(config_name=name),resolve=True)
    (root/output).write_text(yaml.safe_dump(resolved))
    return resolved
train=config('configs/train/grpo','qwen3_mimo','train_config.yaml')
eval_config=config('configs/eval/qwen3','eval_qwen3','eval_config.yaml')
fast=config('configs/train/grpo','qwen3_unpad','fast_config.yaml')
assert fast['actor_rollout_ref']['model']['use_remove_padding']
assert fast['actor_rollout_ref']['actor']['use_dynamic_bsz']
assert fast['actor_rollout_ref']['actor']['ppo_max_token_len_per_gpu']==32768
assert fast['actor_rollout_ref']['actor']['loss_agg_mode']=='seq-mean-token-mean'
fast['actor_rollout_ref']['model']['use_remove_padding']=train['actor_rollout_ref']['model']['use_remove_padding']
for field in ['use_dynamic_bsz','ppo_max_token_len_per_gpu','loss_agg_mode','use_remove_padding']:
    fast['actor_rollout_ref']['actor'][field]=train['actor_rollout_ref']['actor'][field]
# veRL propagates actor performance settings into these shared defaults.
for section in ['ref','rollout']:
    for field in ['log_prob_use_dynamic_bsz','log_prob_max_token_len_per_gpu']:
        fast['actor_rollout_ref'][section][field]=train['actor_rollout_ref'][section][field]
for section, fields in [('critic',['use_dynamic_bsz','loss_agg_mode']),('reward_model',['use_dynamic_bsz'])]:
    for field in fields:
        fast[section][field]=train[section][field]
assert fast==train, 'Unexpected scientific configuration difference'
assert train['actor_rollout_ref']['model']['lora_init_seed']==42
assert train['actor_rollout_ref']['rollout']['temperature']==1.0
assert not train['actor_rollout_ref']['actor']['use_kl_loss']
assert not train['algorithm']['use_kl_in_reward']
assert train['actor_rollout_ref']['actor']['optim']['lr_scheduler_type']=='constant'
assert eval_config['trainer']['val_only'] and eval_config['actor_rollout_ref']['model']['lora_rank']==0
import runpy
build=runpy.run_path('scripts/train/grpo/build_grpo_parquet.py')
tokenizer=AutoTokenizer.from_pretrained(train['actor_rollout_ref']['model']['path'],local_files_only=True)
tools=[t['tool_schema'] for t in yaml.safe_load(Path('configs/tool_config/tau_bench_airline_tools.yaml').read_text())['tools']]
messages=build['build_rows']([0],'train')[0]['prompt']+[{'role':'user','content':'I need help with my reservation.'}]
tokens=tokenizer.apply_chat_template(messages,tools=tools,enable_thinking=False,add_generation_prompt=True)
assert len(tokens)<train['data']['max_prompt_length']
assert tokenizer.decode(tokens).endswith('<think>\n\n</think>\n\n')
print('CONFIG_AND_TOKENIZER_OK prompt_tokens=',len(tokens))
PY
