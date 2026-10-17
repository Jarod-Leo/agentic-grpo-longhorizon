# 实验 B：固定交互预算的 GRPO 动态采样

## 目的与方法来源

在 τ-bench airline 上验证 DAPO 的动态组筛选是否能提高 GRPO 的有效学习信号与最终成功率。这是已有方法的应用验证，不作为原创算法。参考 [DAPO 官方实现](https://github.com/BytedTsinghua-SIA/DAPO) 和仓库 `verl/recipe/dapo/dapo_ray_trainer.py`。

复用 `verl.trainer.main_ppo`、本项目多轮交互、统一评测和 checkpoint 归档。标准 trainer 原本没有调用 group filtering；本次将有界组筛选接入现有循环，不复制独立 trainer。

## 配置

| 项目 | 设置 |
| --- | --- |
| 学生 | 原始 Qwen3-8B，重新初始化 LoRA rank 16 / alpha 32 |
| 初始化与采样种子 | 42 |
| 数据 | 原有 40 train / 10 test；沿用 split.json |
| 用户模拟器 | MiMo `mimo-v2.5`，与 E01 一致 |
| 奖励 | 原始二元 outcome；不加 PRM、LATA 或蒸馏 |
| 每个生成批次 | 4 个任务 × 8 条完整轨迹 |
| 目标更新批次 | 4 个有奖励差异的完整组，最多 32 条 |
| 每次更新尝试最多生成 | 总计 3 批（含第一批），最多 96 条 |
| 训练交互预算 | 6,400 条；被过滤和超过目标组数的轨迹均计入 |
| 优化器更新上限 | trainer steps 上限 200，预算耗尽可更早结束 |
| 学习率与性能配置 | 继承 E01，5e-6、去 padding、动态微批、32768 token/GPU |
| checkpoint | 每 50 次更新尝试及最终保存；先写 SSD，再完整归档 HDD |
| 独立 train eval | 每 100 次更新尝试及最终，40 × 8 |
| 最终 test | 固定预算结束的 checkpoint，10 × 8；普通无辅助提示输入 |

MiMo 官方公告计划于北京时间 2026-10-21 10:00 弃用 V2.5；2026-10-17 已用当前账号确认模型仍在目录。若排队延迟或接口失效，不能在同一实验中静默切换模拟器。切换 V2.6 需要独立标记协议，并重新评测比较基线。

## 采样与更新

1. 从同一个 StatefulDataLoader 取 4 个任务；为每次任务出现分配独立 uid。
2. 当前学生每任务采样 8 条轨迹，计入生成预算；所有轨迹写入环境审计日志。
3. 按 uid 判断 outcome 是否有差异。保留完整混合组，过滤全成功和全失败组。
4. 不足 4 个有效组时继续消费 dataloader，最多共 3 个生成批次，期间不更新模型。
5. 满 4 组时使用前 4 组更新；多出的有效组计入消耗和丢弃统计，不跨模型更新复用。
6. 到达批次数上限或预算边界时，使用实际收集的 1–3 组更新。动态微批损失按实际 minibatch 大小归一化。
7. 没有有效组时跳过 actor 更新，记录一次跳过尝试。预算耗尽时仍执行最终保存和配置的评测。

不把全失败任务永久排除；采样仍遍历整个训练任务集。`global_step` 表示一次收集结束后的更新尝试（可包含零有效组跳过）；另记真实 optimizer update 数，不能把两者混为一谈。

## 预算、恢复与比较

6,400 是完整训练 rollout 的逻辑预算，等于 E01 的 200 × 4 × 8。独立评测和烟测开销单独报告，不算训练 rollout。API tokens、调用次数和 GPU 时间另行记录，因为不同轨迹长度与评测频率会导致总成本不同。

checkpoint 在收集缓冲区清空之后保存：模型、优化器、随机状态、dataloader 状态和动态采样计数共同恢复。若进程在 checkpoint 之间异常中止，恢复时可能重放未保存的采样；这部分真实 API 开销必须计入额外成本，不能宣称总实际调用严格不超过 6,400。

E01 是历史基线；本次 B 的实现和调用时间不同，因此一次种子结果仅是初步证据。有明显收益后需用当前代码补做无筛选对照及更多种子。保持 test 固定用于最终对比，不用 test 挑选 checkpoint 或调整参数。

## 主要观测量

- 所有生成轨迹的单次成功率；筛选后的奖励不能代替总体表现。
- 生成、保留、饱和、超额丢弃的组数和有效比例。
- 真正更新数、跳过次数、每次采样批数、累计轨迹预算。
- 最终 train/test 的 Pass@1、Pass⁴、Pass@4；逐任务成功次数。
- 工具错误、轮数耗尽、API tokens、GPU 时间和每单位交互预算收益。

## 运行入口

- 配置：`configs/train/grpo/qwen3_dynamic_sampling.yaml`。
- 训练：`scripts/train/grpo/run_qwen3_dynamic_sampling.sh`，共用 `run_qwen3.sh`。
- CPU 检查：共用 `cluster_setup/qwen3_shared/check.sbatch`。
- GPU 作业：共用 `cluster_setup/qwen3_shared/run.sbatch`，单张 Pro6000。
- 评测：共用 `scripts/eval/eval_qwen3.sh`；`EVAL_CHECKPOINT_ROOT` 在训练完成后解析最终 checkpoint。
- 产物：`experiments/grpo_dynamic_sampling/`；提交记录包含实际 Job ID。

## 验证与启动记录

2026-10-17：CPU 作业 `209087` 完成，117 项测试全部通过，配置等价检查与 tokenizer 检查通过。覆盖预算计数、完整组筛选、短批次归一化、零有效组时最终评测/保存、主循环实际 actor 调用和计数。五个修改的 veRL 文件通过 Ruff check 和 format --check。

代码冻结为 `experiments/grpo_dynamic_sampling/source-v3`。此前 `source-v1` 的 Hydra 嵌套 searchpath 配置失败，已改为组合公共配置片段；`source-v2` 验证通过后仅作格式调整，形成 v3。

| 作业 | Job ID | 设置与依赖 |
| --- | --- | --- |
| 单卡烟测 | 209091 | 96 条训练轨迹，最终 train eval 40 × 1，保存最终完整 checkpoint，2 小时上限 |
| 正式训练 | 209093 | afterok:209091；6,400 条训练轨迹，36 小时上限，W&B online |
| 最终 test | 209094 | afterok:209093；最终 checkpoint，10 × 8，4 小时上限 |

提交时三者均处于排队/依赖等待，尚无 GPU 验收或实验结果。共享运行入口在退出时执行 summary 验收，失败返回非零，后续依赖作业不能启动。设置完成截止时间为 2026-10-21 01:00 UTC，早于 MiMo V2.5 公告弃用时间；若资源无法及时分配，应重新确认模拟器协议后提交。

正式 checkpoint 归档目录：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/grpo_dynamic_sampling/formal-seed42-budget6400-v1/`。SSD 的 `train/checkpoints/latest_checkpointed_iteration.txt` 记录最终步数，step 目录链接到 HDD，最终评测据此解析 adapter。

清理旧实验中三份已评测较差的 step100 与两份已结束 smoke checkpoint，共释放约 85.49 GB；逐项记录在 `experiments/grpo_dynamic_sampling/checkpoint_cleanup.json`。清理后 HDD 配额使用 253.9/400 GB，剩约 146.1 GB，可容纳本次烟测和正式 checkpoint。自动审批拒绝继续删除未经评测的 step50/150 等恢复点，已保留并请求具体确认；其保留不影响本次实验。
