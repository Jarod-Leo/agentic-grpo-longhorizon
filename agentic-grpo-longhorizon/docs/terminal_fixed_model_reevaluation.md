# 修复终止语义后的模型复评

所有结果直接读取各次评测生成的 `summary.json`；待评测和未验收结果均保留在表中。

## 统一协议

| 配置 | 值 |
| --- | --- |
| name | airline-policy-initial-user-v2 |
| terminal_semantics | first_environment_done_v1 |
| base_model | /projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B |
| simulator_model | mimo-v2.6-pro |
| seed | 42 |
| temperature | 0.7 |
| top_p | 0.9 |
| samples_per_task | 8 |
| train_tasks | 40 |
| test_tasks | 10 |
| assistant_turn_limit | 15 |
| user_turn_limit | 15 |
| rpm | 100 |
| tpm | 10000000 |
| reward_mode | binary |
| optimizer_updates | 0 |
| source_root | /projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/repo/agentic-grpo-longhorizon/experiments/e00_qwen3_baseline/source-terminal-reeval-v1 |
| reused_validation_jobs | [212674, 212684, 212685, 212686, 212687] |

## 结果

| 模型 | 划分 | 状态 | 成功数 | Pass@1 | Pass^4 | Pass@4 | 时长 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E00 · Qwen3-8B base | train | 待评测（[summary](../experiments/e00_qwen3_baseline/terminal-reeval-v1/train/eval/summary.json)） | — | — | — | — | — |
| E00 · Qwen3-8B base | test | 待评测（[summary](../experiments/e00_qwen3_baseline/terminal-reeval-v1/test/eval/summary.json)） | — | — | — | — | — |
| E01 · GRPO step200 | train | 已验收（终止检查通过；[summary](../experiments/e01_vanilla_grpo/train-step200-terminal-v1/eval/summary.json)） | 114/320；40 任务 | 35.62% | 31.11% | 39.96% | 0.283 GPU h |
| E01 · GRPO step200 | test | 已验收（终止检查通过；[summary](../experiments/e01_vanilla_grpo/test-step200-terminal-v1/eval/summary.json)） | 8/80；10 任务 | 10.00% | 10.00% | 10.00% | 0.179 GPU h |
| B · GRPO dynamic sampling update200 / step203 | train | 待评测（[summary](../experiments/grpo_dynamic_sampling/terminal-reeval-v1/train/eval/summary.json)） | — | — | — | — | — |
| B · GRPO dynamic sampling update200 / step203 | test | 待评测（[summary](../experiments/grpo_dynamic_sampling/terminal-reeval-v1/test/eval/summary.json)） | — | — | — | — | — |
| PRM-Lite + LATA · step200 (E02 / historical E05) | train | 待评测（[summary](../experiments/e02_prm_lite_lata/terminal-reeval-v1/train/eval/summary.json)） | — | — | — | — | — |
| PRM-Lite + LATA · step200 (E02 / historical E05) | test | 待评测（[summary](../experiments/e02_prm_lite_lata/terminal-reeval-v1/test/eval/summary.json)） | — | — | — | — | — |
| OPD · step200 (E03 / historical E06) | train | 待评测（[summary](../experiments/e03_opd/terminal-reeval-v1/train/eval/summary.json)） | — | — | — | — | — |
| OPD · step200 (E03 / historical E06) | test | 待评测（[summary](../experiments/e03_opd/terminal-reeval-v1/test/eval/summary.json)） | — | — | — | — | — |
| GRPO + Skill-SD · recovery step200 | train | 待评测（[summary](../experiments/grpo_skill_sd/terminal-reeval-v1/train/eval/summary.json)） | — | — | — | — | — |
| GRPO + Skill-SD · recovery step200 | test | 待评测（[summary](../experiments/grpo_skill_sd/terminal-reeval-v1/test/eval/summary.json)） | — | — | — | — | — |

## Checkpoint

- E00 · Qwen3-8B base: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B`
- E01 · GRPO step200: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3/global_step_200/actor/lora_adapter`
- B · GRPO dynamic sampling update200 / step203: `/projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/repo/agentic-grpo-longhorizon/experiments/grpo_dynamic_sampling/resume-step88-to-update200-seed42-v1/train/checkpoints/global_step_203/actor/lora_adapter`
- PRM-Lite + LATA · step200 (E02 / historical E05): `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e05_prm_lite_lata/formal-seed42-200-v1/global_step_200/actor/lora_adapter`
- OPD · step200 (E03 / historical E06): `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e06_opd/formal-seed42-200-v2/global_step_200/actor/lora_adapter`
- GRPO + Skill-SD · recovery step200: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/grpo_skill_sd/formal-seed42-200-recovery150-v1/global_step_200/actor/lora_adapter`
