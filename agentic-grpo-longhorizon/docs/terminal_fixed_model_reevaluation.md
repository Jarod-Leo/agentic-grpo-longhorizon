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
| E00 · Qwen3-8B base | train | 已验收（终止检查通过；[summary](../experiments/e00_qwen3_baseline/terminal-reeval-v1/train/eval/summary.json)） | 20/320；40 任务 | 6.25% | 0.39% | 13.54% | 0.491 GPU h |
| E00 · Qwen3-8B base | test | 已验收（终止检查通过；[summary](../experiments/e00_qwen3_baseline/terminal-reeval-v1/test/eval/summary.json)） | 1/80；10 任务 | 1.25% | 0.00% | 5.00% | 0.214 GPU h |
| E01 · GRPO step200 | train | 已验收（终止检查通过；[summary](../experiments/e01_vanilla_grpo/train-step200-terminal-v1/eval/summary.json)） | 114/320；40 任务 | 35.62% | 31.11% | 39.96% | 0.283 GPU h |
| E01 · GRPO step200 | test | 已验收（终止检查通过；[summary](../experiments/e01_vanilla_grpo/test-step200-terminal-v1/eval/summary.json)） | 8/80；10 任务 | 10.00% | 10.00% | 10.00% | 0.179 GPU h |
| B · GRPO dynamic sampling update200 / step203 | train | 已验收（终止检查通过；[summary](../experiments/grpo_dynamic_sampling/terminal-reeval-v1/train/eval/summary.json)） | 118/320；40 任务 | 36.88% | 35.04% | 39.43% | 0.347 GPU h |
| B · GRPO dynamic sampling update200 / step203 | test | 已验收（终止检查通过；[summary](../experiments/grpo_dynamic_sampling/terminal-reeval-v1/test/eval/summary.json)） | 8/80；10 任务 | 10.00% | 10.00% | 10.00% | 0.186 GPU h |
| PRM-Lite + LATA · step200 (E02 / historical E05) | train | 已验收（终止检查通过；[summary](../experiments/e02_prm_lite_lata/terminal-reeval-v1/train/eval/summary.json)） | 105/320；40 任务 | 32.81% | 25.21% | 38.54% | 0.314 GPU h |
| PRM-Lite + LATA · step200 (E02 / historical E05) | test | 已验收（终止检查通过；[summary](../experiments/e02_prm_lite_lata/terminal-reeval-v1/test/eval/summary.json)） | 8/80；10 任务 | 10.00% | 10.00% | 10.00% | 0.177 GPU h |
| OPD · step200 (E03 / historical E06) | train | 已验收（终止检查通过；[summary](../experiments/e03_opd/terminal-reeval-v1/train/eval/summary.json)） | 14/320；40 任务 | 4.38% | 0.04% | 11.57% | 0.493 GPU h |
| OPD · step200 (E03 / historical E06) | test | 已验收（终止检查通过；[summary](../experiments/e03_opd/terminal-reeval-v1/test/eval/summary.json)） | 0/80；10 任务 | 0.00% | 0.00% | 0.00% | 0.213 GPU h |
| GRPO + Skill-SD · recovery step200 | train | 已验收（终止检查通过；[summary](../experiments/grpo_skill_sd/terminal-reeval-v1/train/eval/summary.json)） | 115/320；40 任务 | 35.94% | 35.00% | 38.21% | 0.271 GPU h |
| GRPO + Skill-SD · recovery step200 | test | 已验收（终止检查通过；[summary](../experiments/grpo_skill_sd/terminal-reeval-v1/test/eval/summary.json)） | 8/80；10 任务 | 10.00% | 10.00% | 10.00% | 0.177 GPU h |

## Checkpoint

- E00 · Qwen3-8B base: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B`
- E01 · GRPO step200: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3/global_step_200/actor/lora_adapter`
- B · GRPO dynamic sampling update200 / step203: `/projects/_ssd/jiatian001ssd/cabinagentrl/agentic-grpo-longhorizon/repo/agentic-grpo-longhorizon/experiments/grpo_dynamic_sampling/resume-step88-to-update200-seed42-v1/train/checkpoints/global_step_203/actor/lora_adapter`
- PRM-Lite + LATA · step200 (E02 / historical E05): `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e05_prm_lite_lata/formal-seed42-200-v1/global_step_200/actor/lora_adapter`
- OPD · step200 (E03 / historical E06): `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e06_opd/formal-seed42-200-v2/global_step_200/actor/lora_adapter`
- GRPO + Skill-SD · recovery step200: `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/grpo_skill_sd/formal-seed42-200-recovery150-v1/global_step_200/actor/lora_adapter`

## 最终验收与成功任务分布

5个新增GPU作业213220–213224均以退出码0完成；复用的E01作业212686、212687也已完成。6个策略共12项train/test评测全部accepted=true，终止审计通过，没有终止后的额外生成、工具执行或用户API请求。

本次test成功任务逐条核对结果：

| 模型 | 成功任务ID及次数 |
| --- | --- |
| E00 | 任务36：1/8；其余任务0/8 |
| E01 | 任务36：8/8；其余任务0/8 |
| 实验B | 任务36：8/8；其余任务0/8 |
| PRM-Lite＋LATA | 任务36：8/8；其余任务0/8 |
| OPD | 全部任务0/8 |
| GRPO＋Skill-SD | 任务36：8/8；其余任务0/8 |

因此，四个训练模型的test 10%全部来自同一个任务，当前复评没有显示它们能够解决更多未见测试任务。原始模型也只在任务36成功过一次。比较新方法时应基于本次修复后协议，历史高分不能直接作为方法有效性的证据；这次只重新评测既有模型，没有重新训练。
