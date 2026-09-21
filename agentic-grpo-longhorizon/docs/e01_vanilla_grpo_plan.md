# E01：Vanilla GRPO 实施与验证计划

状态：正式作业163430在第75步因MiMo usage=null被错误拒绝而失败，已完成74步且无checkpoint。正在验证修复；用户批准checkpoint每50步、评测每100步。

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
- 公共训练入口：`scripts/train/grpo/run_qwen3.sh`；公共评测入口：`scripts/eval/eval_qwen3.sh`。评测 adapter 时设置 `ADAPTER_PATH`；三步验证编排为 `scripts/train/grpo/verify_qwen3.sh`。
- 历史 `scripts/eval/eval_e00_qwen3.sh` 已收敛为公共评测入口的兼容壳，默认使用 v2 协议。历史 E00 原始脚本与配置保存在 `experiments/e00_qwen3_baseline/source_before_protocol_v2/`；不使用新脚本重新解释旧协议结果。
- 公共 Slurm 壳：仓库根目录 `cluster_setup/qwen3_shared/run.sbatch`；CPU 预检壳为同目录 `check.sbatch`。作业通过现有 `setup/activate.sh` 加载环境及 `~/.config/cabinagentrl/mimo.env`，不把密钥写入配置或命令。
- 不可变源码快照：`scripts/train/grpo/snapshot_qwen3_source.py <新目录>`，同时生成 `source_manifest.json`。E00 固定使用 `experiments/e01_vanilla_grpo/source-ready-v1/`，当前 E01 固定使用 `experiments/e01_vanilla_grpo/source-ready-v3/`；运行中的快照禁止修改。

初版 CPU 预检作业 **162645** 已成功：27 项测试通过；训练与评测配置解析成功；使用实际 Qwen3 tokenizer，完整规则、工具与示例用户消息为 **3929 tokens**，非思考后缀正确。此前发现并修复了快照漏带旧划分 fixture、Hydra secondary config 不允许重设 searchpath，以及第三方 stdout 日志污染 YAML 解析的问题。修改的四个 veRL Python 文件已通过 Ruff check 与 format --check。

GPU 提交记录：

| 作业 | Job ID | 工作 | 输出目录（相对主项目） |
| --- | --- | --- | --- |
| 新 E00（已通过） | 162649 | 原始模型，v2 输入，train 40×8 | `experiments/e00_qwen3_baseline/protocol-v2-ready-v1/eval/` |
| E01 readiness（重提） | 162772 | continuous 3 步 → 从 step 2 resume → continuous step 3 评测 | `experiments/e01_vanilla_grpo/readiness-v4/` |

E00 作业 162649 已成功完成，320 条轨迹验收全部通过；新 E01 作业在确认基线通过后直接提交。两者均为单 PRO 6000、最长 2 小时的正式 sbatch 作业。提交记录见 `experiments/e01_vanilla_grpo/submissions.json`。这些是验证作业，提交成功不等于训练/恢复/评测已全部通过，也不代表已开始正式预算训练。

每个阶段的关键文件：`run.json`（划分与模式）、`run.log`、`eval_trajectories.jsonl`、`tool_audit.jsonl`、`api.jsonl`、`metrics.jsonl`、`summary.json`、`report.md`。三步训练与恢复在 `continuous/` 和 `resume/`；恢复一致性检查为顶层 `readiness.json`；固定 checkpoint 评测为 `eval_step3/summary.json`。只有各阶段验收通过，才能报告完整验证成功。`learning_signal_verified=false` 时还不能宣称已验证有效学习。

查看作业状态：

```bash
squeue --me
sacct -j 162649,162772 --format=JobID,State,Elapsed,ExitCode
```

Git 同步目标：`https://github.com/Jarod-Leo/agentic-grpo-longhorizon.git`。当前执行目录为部署副本；Git 工作树位于 `experiments/e01_vanilla_grpo/github-worktree/`，代码同步使用 `e01-vanilla-grpo` 分支，原始轨迹、缓存和 checkpoint 不推送；保留小型验收汇总、提交记录及文档。

启动补充：初次 E00 使用空 CUDA 缓存，观察到 NCCL 初始化期间持续生成编译缓存。原待运行 E01 作业 162650 已取消，替换为 162652；新快照允许显式传入 CUDA_CACHE_PATH，并复用历史已验证缓存，减少连续训练、恢复及评测三个进程的重复编译。运行中的 E00 快照保持原样。


## 温度元数据故障修复与重提

原 E01 作业 162652 在完成首批 32 条轨迹后、首次 actor 更新前失败，耗时 7 分 8 秒，未保存 checkpoint。错误为 `dp_actor.update_policy` 读取 `data.meta_info["temperature"]` 时触发 KeyError。

根因：异步 `AgentLoopWorkerBase.generate_sequences` 使用 rollout 配置构造采样温度，但 `_postprocess` 输出未携带该温度。旧训练的 `compute_log_prob` 或 KL reference pass 会顺带提供此字段；E01 的 bypass + 无 KL 路径跳过二者，因此缺少元数据。

最小修复：在公共异步 rollout 返回批次上设置 `output.meta_info["temperature"] = sampling_params["temperature"]`。这同时保留训练温度与评测实际使用的温度；不修改 actor 的严格读取、不设隐式默认值，也不改变采样、奖励或算法配置。

新增回归覆盖异步 rollout、批次 concat/union、bypass 处理及实际 CPU actor 更新，使用非 1.0 温度检查传递是否正确，核对有限梯度与真实参数变化；另检查评测温度。CPU 预检 **162768** 完成，**29 项测试通过**，配置与实际 tokenizer 检查通过，修改文件的 Ruff check / format --check 均通过。

新作业 **162772** 使用 `source-ready-v3`，相比上一快照仅变更 `verl/verl/experimental/agent_loop/agent_loop.py` 和对应回归测试。失败运行没有可恢复的 checkpoint，因此从原始模型和 seed 42 的 LoRA 初始化重新执行三步验证；输出使用 `readiness-v4`，历史失败记录保留。此提交仍不代表 GPU 完整链路已通过。

已通过的 E00 v2 结果：40 train 任务、320 条轨迹、56 次成功；Pass@1=17.50%，Pass^4=2.86%，Pass@4=38.04%。验收汇总和报告位于 `experiments/e00_qwen3_baseline/protocol-v2-ready-v1/eval/`。此次修复仅补充训练消费的元数据，沿用已通过的基线。

节点补充：温度修复后的第一次提交 162771 被分配到 `gpu-pro6000-3`，该节点的 tokenizer 加载器未能识别已有 HDD 模型目录，触发 HFValidationError；未生成轨迹或执行更新。当前提交 **162772** 使用同一 `source-ready-v3`，指定 E00 已验证可读取模型的 `gpu-pro6000-11`，输出改为 `readiness-v4`。此资源调整不改变实验配置或代码；节点满载时由 Slurm 排队。


## 启动入口简化

工作目录中的 `run_qwen3.sh` 已精简为 15 行，`eval_qwen3.sh` 为 8 行：入口只加载公共准备函数、选择训练或评测配置并调用 veRL。环境变量、数据准备、缓存、日志和退出验收集中维护在 `scripts/train/grpo/qwen3_runtime.sh`，训练与评测共同复用。这里是职责拆分，必要的运行管理逻辑仍然保留。

激活环境并取得 Slurm 资源后，调用方式为：

```bash
bash scripts/train/grpo/run_qwen3.sh
bash scripts/eval/eval_qwen3.sh
```

额外参数继续传给 Hydra；`RUN_DIR`、`RESUME_FROM`、`ADAPTER_PATH`、`STEPS`、`SAMPLES` 等已有环境参数保持兼容，旧 `run_qwen3.sh train|eval` 调用也仍可使用。训练超参数继续在 YAML 中修改。

本次通过 Bash 语法检查和 8 项模拟入口检查，覆盖训练、评测、旧调用方式、恢复、adapter、训练失败和汇总失败的退出处理；未增加 GPU 测试作业。已提交的 162772 仍使用原 `source-ready-v3` 固定快照，不改变该作业脚本，也不因这次等价整理重新提交。


## 2026-09-20 加速验证

作业 162772 的连续三步训练已完成，continuous/summary.json 的 accepted=true；恢复和评测尚未完成。前三步 rollout 分别约 313/391/250 秒，actor 更新约 250/250/252 秒，checkpoint 保存各约 51 秒。564 次 MiMo 请求均为 HTTP 200，单请求平均 API 延迟 5.06 秒、限流等待 7.75 秒；并发请求的等待之和不能当成串行墙钟耗时。用户已确认实际额度就是 100 RPM / 1000 万 TPM，保持该限流配置，不通过增加 worker 绕过额度。

当前 remove-padding 为 false，agent loop 将 prompt/response 补齐到 8192+12288=20480 token，而各步平均有效总长度约 5646/6284/5574。使用 veRL 原生 `model.use_remove_padding=true` 可跳过 padding 计算；不能把 token 长度比例直接当成整体加速比。训练显存峰值约 76.3 GiB，目前不同时增加 micro batch 或关闭 checkpointing，以便隔离效果。

新增 `configs/train/grpo/qwen3_unpad.yaml` 复用 ppo_trainer/qwen3_common。训练入口通过 `QWEN3_CONFIG` 选择配置，默认仍为 qwen3_mimo。Hydra compose 检查确认两个配置仅 model.use_remove_padding 不同；shell 语法检查通过。没有修改 veRL 实现，也不改在跑快照。

单步 GPU 验证作业 **162833**：单 PRO6000、30 分钟上限、4 tasks × 8 rollouts、seed42、从原始模型初始化；`afterok:162772` 且依赖失效自动取消，避免与当前验证争用 MiMo 额度。冻结源码 `experiments/e01_vanilla_grpo/source-unpad-v2`，输出 `experiments/e01_vanilla_grpo/unpad-smoke-v1/train`。source-unpad-v1 是未提交的配置草稿，v2 是实际提交版本。现阶段仅确认提交成功，GPU 兼容性、梯度、显存和耗时尚待验证；默认训练配置暂不切换。通过后对比 actor/update 耗时及实际有效 token 数，保留 rollout 随机性对端到端比较的限制。

GPU 的短时采样覆盖恢复初始化阶段：曾见 0%（加载/编译）及 100%（约 58 W、memory utilization 0%）。这些瞬时值不能证明有效算力满载；判断加速以阶段耗时和更新吞吐为准。


## 动态微批测速：32768 → 24576 → 20480

用户更新了执行顺序：先测 32768，只有明确的 GPU OOM 才降到 24576，再 OOM 才降到 20480；首个成功档位即停止，不再按显存占用从低向高扫描。旧作业 162833 已取消。原 162772 已成功完成连续训练、恢复检查和 320 条 train 评测，验收均通过；step-3 独立评测 Pass@1=17.1875%、Pass^4=2.9286%、Pass@4=34.5714%，不能据单次小规模验证推断效果优劣。

`qwen3_unpad.yaml` 开启 remove-padding 与 dynamic microbatch，初始 `actor.ppo_max_token_len_per_gpu=32768`，保留 FlashAttention2。框架依据实际 token 数和长度负载组织微批；预算是分组参数，不保证各微批实际 token 总数绝不超过该值。`loss_agg_mode=seq-mean-token-mean` 保持原 microbatch=1、token-mean 累积时每条轨迹等权，变长轨迹损失与梯度一致性有 CPU 测试覆盖。32 条轨迹、训练 seed、优化器更新次数及科学配置保持原设定。

共用测速入口 `scripts/train/grpo/benchmark_qwen3.sh` 调用现有训练脚本，不复制训练逻辑。每档全新初始化、1 步，最多三档；单档超时 25 分钟，整个 Slurm 作业上限 2 小时。只有 CUDA OOM 明确日志触发回退；API、CPU 内存、正确性和超时问题停止任务。GPU 每秒采样，汇总 rollout、actor update、step、checkpoint、总墙钟、启动及其他开销、有效 token 数、设备显存峰值与 PyTorch allocated/reserved 峰值。`performance.json`/`performance.md` 保存结果，原始采样在每档 `gpu.csv`。

SSD 125.7/150 GB，用户明确允许清理本次测速的完整 checkpoint。仅在该档成功验收后，由 `CLEAN_BENCHMARK_CHECKPOINTS=1` 删除该新输出目录中的 model/optim/extra_state 三个 rank-0 文件，保留 LoRA adapter、配置、日志、测速结果及删除记录；既有 E01 checkpoint 不动。验收后清理意味着本次测速产物只保留 adapter，不能当成完整训练恢复点。失败档保留现场且不会继续占用 checkpoint 保存空间，除非发生保存阶段故障（此类错误不会触发 OOM 回退）。

验证：四种轻量模拟检查覆盖首档成功、一次/两次 OOM 回退、非 OOM 停止及 adapter 保留。CPU 预检首次运行 30 tests passed，但配置差异断言未计入 veRL 自动继承的性能字段，已修正预检并使用新快照重跑；未修改 veRL 生产实现。GPU 实测结果以新作业生成的 performance 文件为准，提交成功不代表已验证提速。

CPU 预检 162930 已完成：30 tests passed，配置/tokenizer 检查通过。GPU 测速作业 **162932** 已提交，快照 `source-dynamic-v2`，输出 `dynamic-smoke-v1`，最大 2 GPU 小时。两项前置作业均已成功完成，Slurm 不再接受其历史依赖，核验后直接提交。


## 正式 E01：200 步，周期评测与 HDD 归档

用户明确将 E01 正式预算定为 200 步，替代本实验此前未冻结的 20/30/50 步候选；其他实验预算未据此自动确定。使用基础 Qwen3-8B 与相同 seed42/LoRA 初始化，采用已验收的 remove-padding、动态微批32768、轨迹等权损失，保留原 batch=4、rollout n=8、lr=5e-6。40 train / 10 test 划分不变。

`qwen3_formal.yaml`：200步、20 epochs（40个训练任务每轮10个batch）、save_freq=test_freq=100，训练前不额外评测。第100和200步均用40个train任务×8次采样评测，test保持未见。预期训练轨迹6400条、周期评测640条，总计7040。训练与评测共用veRL，轨迹新增validate标志，汇总分别验收与报告，避免同一步训练/评测混成GRPO分组。通用入口根据总步数配置足够epochs，元数据保存评测与checkpoint计划。

归档目录：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v1`。遵循集群活跃IO使用SSD规则，完整checkpoint先写本次SSD输出；模型、优化器、随机状态、scheduler、data loader、adapter全部写完后，通过可选 `trainer.checkpoint_archive_dir` 复制到HDD临时目录，再重命名为正式目录。复制成功后释放本次SSD副本，原位置保留指向归档的链接，兼容既有发现/恢复和验收逻辑。复制失败时保留SSD完整checkpoint并停止，不覆盖已有HDD目录。两份完整checkpoint均长期保留，既有验证checkpoint不动。要再次训练时可按现有恢复入口显式读取对应归档。

Slurm：单PRO6000、gpu-pro6000-11（已验证模型可访问）、36小时上限；不更改共用run.sbatch，提交时覆盖时限。按单步约400秒估算，训练约22.2 GPU小时，另加两次评测、初始化和归档；实际轨迹变长可能增加耗时。MiMo保持100RPM/1000万TPM。正式训练不在OOM后自动改变配置；测速回退入口仅用于测速。

验证包括：原有用例、同一步训练/评测隔离与缺失拒绝、外部checkpoint路径与完整性、归档保留数据/本地链接、复制失败时SSD副本保留。框架Ruff check和format检查通过。正式提交使用冻结快照，运行后不修改其代码。

最终CPU预检 **163429**：33 tests passed，正式配置和7040轨迹元数据检查通过。正式作业 **163430** 已提交；冻结源码 `source-formal-v2`，SSD输出 `experiments/e01_vanilla_grpo/formal-seed42-200-v1/train`。


## MiMo失败修复及恢复checkpoint频率

详见 `e01_mimo_failure_analysis.md`。MiMo允许usage=null；有效回复仍使用原文本，费用标为未知。异常回复增加有限重试和安全结构化诊断。正式配置更新为200步、checkpoint每50步、评测每100步，预期训练6400+评测640=7040条轨迹不变。HDD保留50/100/150/200完整checkpoint；新正式训练必须从基础模型开始，不能声称恢复原第74步。
