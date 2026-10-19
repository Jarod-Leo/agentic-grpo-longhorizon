# 修复执行环境后重新训练 GRPO

## 目标与当前授权

用户要求按已讨论的路线，从原始Qwen3-8B重新训练GRPO。总目标为200次更新；先执行25次更新的短程验证，检查真实成功轨迹、训练信号和完整checkpoint，再从同一新训练状态继续。旧E01以及其他旧训练模型不作为初始化。实验C方案保留在 [实验C文档](grpo_local_branch_experiment_c.md)，本轮不接入局部分支损失。

## 重训前发现并处理的问题

1. 已修复首次环境终止后继续生成、调用工具和再次评分的问题，使用 `first_environment_done_v1`。
2. 本轮预检确认工具返回曾被veRL默认 `max_tool_response_length=256` 按字符中部截断；这个截断实际作用于策略输入，不只是日志展示。预订航班、乘客、付款方法等中间字段可能被删除。通过共用配置显式提高到65536字符，并记录到运行元数据；整体上下文和响应token预算仍然有效。
3. CPU回归检查验证真实工具执行路径能保留中间字段，并复用首次终止奖励及汇总检查。GPU工作全部通过Slurm。

上一轮全模型复评已解决终止问题，但仍采用256字符工具返回限制。其分数只能代表当时协议下的结果，不能直接作为本轮训练输入修正后的基线。先使用新设置重新评测原始Qwen3-8B的train/test；本轮不重复评测全部旧模型。

## 训练配置

| 项目 | 配置 |
| --- | --- |
| 初始化 | 原始Qwen3-8B，新的LoRA，rank16、alpha32、初始化seed42 |
| 数据 | 原40个train任务；10个test任务独立保留，无dev |
| 算法 | 普通GRPO，原始二元任务结果；不加PRM、LATA、OPD、Skill-SD或分支loss |
| 根轨迹 | 每步4任务×8条=32条 |
| 学习率 | 5e-6，constant；PPO epoch1；无KL loss |
| 性能配置 | 去padding、动态微批、max_token_len_per_gpu=32768；OOM时按既定24576、20480降档 |
| 用户模拟器 | MiMo v2.6 Pro，100 RPM、1000万TPM |
| 工具返回上限 | 65536字符；保留同一15轮assistant/user限制 |
| 上下文预算 | prompt8192、response12288 token；模型上限24576 |
| 训练采样 | temperature1、top_p1 |
| 独立评测 | 每任务8次，temperature0.7、top_p0.9 |
| 当前阶段 | 从base训练25步，第25步保存完整checkpoint并评测train |
| 后续阶段 | 验收后恢复到累计200步；每50步保存、每100步评测train，最终test |

25步属于这次新训练的前25步，不是额外的旧模型续训。恢复时必须匹配模型、优化器、调度器和数据状态。短程验收不要求成功率达到任意高阈值，但应出现可解释的奖励差异、有限梯度和正确的实际业务操作；不能仅凭进程退出码或loss下降判断学会任务。

## 存储与运行

提交前查得SSD使用74.1/150GB，HDD使用373.6/400GB。当前25步阶段只保存一份约17GB完整checkpoint到HDD，保留既有产物；正式200步阶段的多份checkpoint需要在验收时重新核对容量与存储安排，不把剩余HDD空间当成足够容纳四档。

- CPU入口：共用Slurm CPU壳调用 `scripts/train/grpo/check_qwen3_observation.sh`。
- Base评测：共用GPU壳调用 `scripts/eval/eval_qwen3_splits.sh`，依次train/test。
- 25步训练：共用GPU壳调用 `scripts/train/grpo/run_qwen3.sh`，选择 `qwen3_formal` 配置，由环境变量设置25步与保存/评测频率。
- 所有作业提交前冻结源码；每个运行目录记录 `submission.json` 与含Job ID的日志。
- WandB沿用 `carrortsky-nanyang-technological-university-singapore/Agentic-longhorizon`，凭据通过现有受保护文件注入，不写入文档或提交记录。

## 验收后继续工作的检查清单

- [x] CPU检查通过，工具中间字段确实进入策略上下文。
- [ ] 新协议base train/test结果完整、验收通过。
- [ ] 新GRPO完成25步、完整checkpoint保存并可供恢复。
- [ ] 检查梯度/奖励差异、工具错误、转人工与首次终止行为，以及真实成功轨迹。
- [ ] 确定后续存储安排，恢复到累计200步并按计划保存、评测。

## 本次提交记录（2026-10-19 UTC）

| 作业 | 任务 | 提交后状态 |
| --- | --- | --- |
| 213638 | CPU回归与配置检查 | COMPLETED，38秒，23项测试通过，配置与3929-token提示检查通过 |
| 213639 | 原始Qwen3-8B train/test基线 | PENDING，等待GPU资源 |
| 213640 | 从base开始GRPO前25步 | PENDING，依赖213639成功结束 |

冻结源码位于 `experiments/e01_vanilla_grpo/source-retrain-observation-v1`。训练输出位于 `experiments/e01_vanilla_grpo/retrain-observation-fixed-seed42-v1`，基线输出位于 `experiments/e00_qwen3_baseline/observation-fixed-v1`。作业依赖失败时不会启动后续训练。

第25步完整checkpoint归档到 `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/retrain-observation-fixed-seed42-v1/global_step_25`。当前只提交25步阶段，尚未提交续训到200步的作业。

边界说明：当前15轮assistant上限会在第15轮工具调用执行前结束，因此最多执行14轮工具动作；本轮保留已有轮数协议。极端大型工具返回仍可能耗尽12288-token累计响应预算，需要在短程轨迹验收中检查，不能把65536字符上限理解为无限上下文。
