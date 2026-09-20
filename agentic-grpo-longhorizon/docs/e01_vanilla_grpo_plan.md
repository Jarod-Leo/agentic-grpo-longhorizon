# E01：Vanilla GRPO 实施与验证计划

状态：方案已确认；先完成三步训练、恢复与评测验证，正式训练步数待全实验预算冻结。

## 实验边界

沿用 `experiments/sft_collect_airline/split.json` 的 40 个 train / 10 个 test 任务，不设 dev。训练与研发检查仅使用 train；全实验配置冻结后统一评测 test。原 E00 的 30 train / 10 dev 标签及原始结果保留，可按这 40 个 train 重新汇总，但不能与修复后的输入协议混为一组结果。

E01 从官方 Qwen3-8B 开始，二值 outcome reward，GRPO 组内标准化；不启用 PRM、LATA、KL、熵奖励或 OPD。先完成公共输入修复和新基线，再验收三步训练、断点恢复、checkpoint 评测。正式 E01 从相同基础模型和 LoRA 初始化重新开始，不继承验证阶段权重。

## 必须先修复的输入问题

历史 E00 的 system prompt 只有日期说明，未包含 airline WIKI；`env.reset()` 返回的首条用户消息没有交给策略模型。新协议 `airline-policy-initial-user-v2`：

1. system prompt 包含完整 airline WIKI 和原有日期说明。
2. 保存 `env.reset().observation`，在 veRL 现有 pending hook 中将其作为首条 user message 注入一次；不额外调用用户模拟器生成开场白，不泄露任务指令或预期动作。
3. 检查实际 tokenized prompt 中的完整工具 schema、规则、用户消息和 Qwen3 非思考模式。
4. 保持 15 assistant / 15 user / 30 environment turns 及现有终止和二值奖励语义。首条 user 属于 prompt；后续 user、tool、padding 不参与 actor loss。
5. 训练与评测共用同一 agent loop，正常退出与异常退出均释放 interaction。

修改前确认无在跑作业，保存历史相关源码、配置及 SHA256；新结果使用新目录，不覆盖旧 E00。

## 公共入口与配置

优先复用 veRL `main_ppo`、`val_only`、LoRA adapter 加载和 checkpoint 恢复能力。新增差异配置与薄入口，不复制一套实验运行程序。公共逻辑置于 `src/`、`scripts/`、`configs/`，Slurm 壳仅负责资源、环境和调用。原始模型与训练 adapter 使用同一评测程序。

汇总器由实际任务集合和每题重复次数校验完整性，不硬编码旧 30/10 划分或 320 条。统一报告 task-macro Pass@1（成功率）、Pass^4（四次全成功可靠性）和 Pass@4（四次至少一次成功）；复用 `src/evaluation/pass_metrics.py`。另报工具调用、API、终止原因及 GPU 时间。失败或缺失轨迹不应被默默排除并作为合格结果。

## E01 固定配置

| 项目 | 配置 |
| --- | --- |
| 基础模型 | 官方 Qwen3-8B，BF16，非思考模式 |
| 参数更新 | LoRA rank 16、alpha 32、all-linear；AdamW，lr 5e-6，constant，无 warmup |
| 算法 | GRPO，norm_adv_by_std=true，binary outcome；KL/entropy=0 |
| 一步采样 | 4 任务 × 8 trajectories = 32；mini-batch 4、micro-batch/GPU 1、PPO epochs 1 |
| 训练采样 | temperature=1.0、top_p=1.0、top_k=-1 |
| 评测采样 | temperature=0.7、top_p=0.9、top_k=-1、每题 8 次 |
| 长度与并发 | prompt 8192、response 12288、model 24576、max_num_seqs=12、GPU utilization=0.5 |
| 随机性 | data / vLLM / LoRA init seed=42；记录初始 adapter hash |
| 保存 | 验证阶段每步保存模型、optimizer、scheduler、RNG、data loader |
| 资源 | 单节点单张 pro6000；MiMo 经现有 API 使用，无 simulator GPU |

LoRA 初始化新增可选种子字段，默认 None 保留旧行为；仅在初始化作用域内设种子，不扰动外部随机状态。

## 验证与作业顺序

1. 轻量测试：规则及初始用户消息、恰好一次注入、任务隔离、异常清理、response mask、工具 schema、指标已知值。API 凭据仅由 `setup/activate.sh` 加载。
2. 新 E00：官方原始模型，train 40×8=320，使用修复后的公共输入与评测入口。
3. E01 continuous：连续三步，共 96 条训练轨迹，保存 step 1/2/3。
4. E01 resume：从 continuous step 2 在独立目录恢复，`total_training_steps=3` 不变，仅补第三步（32 条额外轨迹）。核对完整 checkpoint、恢复的 optimizer/scheduler/RNG/data loader、下一批任务与 continuous step 3 对应；MiMo 存在随机性，不要求恢复分支结果逐位相等。
5. 使用同一评测入口评测 continuous step 3 adapter，train 40×8=320，不择优选 resume 分支。
6. 验收：完整轨迹、有限奖励/梯度/权重、真实参数更新、完整可恢复状态、LoRA 导出和加载成功。全失败或全成功组的零 GRPO advantage 是合法现象；连续三步没有混合成功组时报告学习信号不足，不修改奖励掩盖问题。

提交前刷新 GPU/QoS/账户和存储检查；代码使用不可变快照。新基线与 E01 验证分别使用单 pro6000、最长 2 小时的 sbatch 作业，保存 Job ID、资源、服务/API、rollout、训练与 Slurm 日志。完成初期检查和提交即可停止监控，无需持续等待实验结束。

## 总预算及正式实验

总预算 80 GPU 小时：预检 10、评测和失败预留 15、正式训练 55。主矩阵 E00/E01/E05(PRM+LATA)/E06(OPD)/E07(GRPO+OPD)，E01/E07 第二种子固定纳入。OPD 实测速度后统一确定正式 20/30/50 步预算，避免用 test 或验证阶段回报挑选配置。正式结果保存种子、划分、代码哈希、模型与 checkpoint；实验跑通完成后按项目要求推送 GitHub。

## 已实现入口与运行记录（2026-09-20）

- 公共配置：`configs/train/grpo/qwen3_common.yaml`；训练配置 `qwen3_mimo.yaml`，评测差异配置 `configs/eval/qwen3/eval_qwen3.yaml`。
- 公共训练入口：`scripts/train/grpo/run_qwen3.sh train`；公共评测入口：`scripts/eval/eval_qwen3.sh`。评测 adapter 时设置 `ADAPTER_PATH`；三步验证编排为 `scripts/train/grpo/verify_qwen3.sh`。
- 历史 `scripts/eval/eval_e00_qwen3.sh` 已收敛为公共评测入口的兼容壳，默认使用 v2 协议。历史 E00 原始脚本与配置保存在 `experiments/e00_qwen3_baseline/source_before_protocol_v2/`；不使用新脚本重新解释旧协议结果。
- 公共 Slurm 壳：仓库根目录 `cluster_setup/qwen3_shared/run.sbatch`；CPU 预检壳为同目录 `check.sbatch`。作业通过现有 `setup/activate.sh` 加载环境及 `~/.config/cabinagentrl/mimo.env`，不把密钥写入配置或命令。
- 不可变源码快照：`scripts/train/grpo/snapshot_qwen3_source.py <新目录>`，同时生成 `source_manifest.json`。两个 GPU 作业固定使用 `experiments/e01_vanilla_grpo/source-ready-v1/`；运行中的快照禁止修改。

CPU 预检作业 **162645** 已成功：27 项测试通过；训练与评测配置解析成功；使用实际 Qwen3 tokenizer，完整规则、工具与示例用户消息为 **3929 tokens**，非思考后缀正确。此前发现并修复了快照漏带旧划分 fixture、Hydra secondary config 不允许重设 searchpath，以及第三方 stdout 日志污染 YAML 解析的问题。修改的四个 veRL Python 文件已通过 Ruff check 与 format --check。

GPU 提交记录：

| 作业 | Job ID | 工作 | 输出目录（相对主项目） |
| --- | --- | --- | --- |
| 新 E00 | 162649 | 原始模型，v2 输入，train 40×8 | `experiments/e00_qwen3_baseline/protocol-v2-ready-v1/eval/` |
| E01 readiness | 162650 | continuous 3 步 → 从 step 2 resume → continuous step 3 评测 | `experiments/e01_vanilla_grpo/readiness-v1/` |

E01 使用 `afterok:162649` 依赖，基线验收失败时不启动。两者均为单 PRO 6000、最长 2 小时的正式 sbatch 作业。提交记录见 `experiments/e01_vanilla_grpo/submissions.json`。这些是验证作业，提交成功不等于训练/恢复/评测已全部通过，也不代表已开始正式预算训练。

每个阶段的关键文件：`run.json`（划分与模式）、`run.log`、`eval_trajectories.jsonl`、`tool_audit.jsonl`、`api.jsonl`、`metrics.jsonl`、`summary.json`、`report.md`。三步训练与恢复在 `continuous/` 和 `resume/`；恢复一致性检查为顶层 `readiness.json`；固定 checkpoint 评测为 `eval_step3/summary.json`。只有各阶段验收通过，才能报告完整验证成功。`learning_signal_verified=false` 时还不能宣称已验证有效学习。

查看作业状态：

```bash
squeue --me
sacct -j 162649,162650 --format=JobID,State,Elapsed,ExitCode
```

Git 同步目标：`https://github.com/Jarod-Leo/agentic-grpo-longhorizon.git`。当前执行目录为部署副本；Git 工作树位于 `experiments/e01_vanilla_grpo/github-worktree/`，代码同步使用 `e01-vanilla-grpo` 分支，运行产物、缓存和 checkpoint 不推送。
