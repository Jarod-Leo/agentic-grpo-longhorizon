# 验证成功与纠错 SFT、GRPO、关键动作局部优势的最小对照

## 目标与授权

2026-10-20，用户要求执行已讨论的最小 A/B/C 实验。目标是检验：经过真实环境验证的业务成功与纠错监督能否补齐当前 GRPO 的写操作能力，以及在相同 SFT 起点上增加局部信用分配是否进一步改善任务成功率。

所有方法沿用当前首次终止修复、65536 字符工具返回、Qwen3-8B、40 train / 10 test、MiMo v2.6 Pro 和共享 veRL 训练/评测入口。测试集不参与示范收集、分支规则设计或参数选择。

## 实验矩阵

| 条件 | 初始化 | 更新目标 | 作用 |
| --- | --- | --- | --- |
| A | 原始 Qwen3-8B | 普通 GRPO，200 步 | 复用修复环境后的既有训练 |
| B | 原始 Qwen3-8B，经验证成功/纠错 SFT 的 adapter | 普通 GRPO | 检验额外监督的训练方案收益 |
| C | 与 B 完全相同的 SFT adapter | GRPO + 关键动作局部 PG | 相对 B 检验局部信用分配 |

B 与 C 分别创建新的、同配置优化器；SFT 权重保持一致，不把 A 的优化器与 SFT 权重混用。先做 25 步机制对照，成功后才扩至 200 步。25 步结果只与同预算 B/C 比较，不能当作与 A 200 步的等预算结论。

额外教师采集、SFT 和分支开销分别记录；等更新数比较不等于等交互成本。最终评测使用原始单轨迹执行，不在测试时调用教师或分支搜索。

## A 的实际状态

新训练 `e01_vanilla_grpo/retrain-observation-fixed-seed42-200-resume50-v1` 已完成累计 200 步，汇总 accepted=true；step200 固定 train 评测 104/320（32.5%）。

检查时，已完成的 `old-step200-observation-fixed-v1/test` 是旧 E01 adapter 的同协议复评（8/80），不是新重训 adapter 的测试。新模型 test 补评通过原冻结源码提交为作业 **215803**，输出 `experiments/e01_vanilla_grpo/test-retrain-observation-fixed-v1`。不重跑 A 的训练。

## B：验证数据与 SFT

旧 SFT 采集器使用独立 wrapper 和旧协议；本实验不直接复用旧成功标签。复用当前 TauBenchAgentLoop、工具适配器、交互与首次终止评分，只扩展教师生成及监督数据导出。

- 教师：现有 MiMo v2.6 Pro API；与用户模拟器分开保存对话，共享调用额度限制。教师只看到 policy、工具 schema 与可见对话，不读取隐藏用户任务或参考动作。
- 从头示范：教师通过同一执行路径完成训练任务。
- 纠错示范：学生产生前缀，在首个可恢复工具错误之后由教师继续；保留真实错误和工具反馈。仅监督教师后缀，学生前缀作为上下文。
- 只接收原始环境二元成功、无终止后调用、无截断污染的轨迹。业务合规抽查与终局成功分别记录，不能声称二元奖励证明所有政策约束。
- 初始可行性验证：train 任务 0、1、4、6、13、45，各 2 条；覆盖预订、取消、行李、航班更新、转人工和补偿。此选择不使用 test。
- 可行性通过后，完整采集预算：40 train × 4 条从头示范，40 train × 2 条学生纠错尝试。失败尝试同样计入成本；不无限补采到成功。
- SFT 复用 veRL FSDP SFT trainer，Qwen3-8B、LoRA rank16/alpha32、seed42、lr1e-5；正式 2 epochs，不用 test 选 epoch。先做短 smoke 验证 assistant/后缀 mask 和 adapter 可加载。

若教师不能解决写操作任务，或纠错数据没有真实成功样本，则记录这一阶段的负结果，不把空数据/只读成功冒充已完成 B。先定位执行或教师能力问题，再调整有限预算。

## C：首版局部优势

复用现有快照、恢复和 suffix rollout；局部项只监督原始 root 轨迹中的候选动作，以保持原始训练批次和每步一次 optimizer update。

每个任务组仅选一个 root rollout，每条 root 最多一个分支点。选点仅依赖动作生成前的历史：上一工具返回错误，或上一成功工具为 `get_reservation_details`、`search_direct_flight`、`search_onestop_flight` 时，选择随后的第一个 assistant 决策。这样覆盖获取业务信息后的选择和错误恢复机会；不读取 root 已生成的动作来筛选分支，也不强制下一步一定是写操作。分支必须保留动作前数据库、用户历史、策略前缀以及剩余预算。

候选 0 为 root 实际采样的动作 token，不重新采样替换；其余 K−1 个动作由同一冻结策略在相同前缀生成。每个动作固定后执行 M 次独立后缀：

$$
\widehat Q_k=\frac{1}{M}\sum_{m=1}^{M}R_{k,m},
\qquad
A_{\mathrm{local}}=\widehat Q_0-\frac{1}{K-1}\sum_{k=1}^{K-1}\widehat Q_k.
$$

$$
L=L_{\mathrm{GRPO}}+0.25L_{\mathrm{local}}.
$$

局部 clipped PG 使用原 root 动作的固定 old log-prob；只 mask 该动作 token。前缀、后续续写及额外候选不进入局部损失。两项 loss 分别计算后相加，不把两个 advantage 直接相加后统一 clipping。局部项按实际有效分支动作平均，微批切分不能改变归一化。

首轮 K=4、M=2 验证执行；M=8 冻结诊断检查候选回报差异及两半续写稳定性。远程用户模拟器的随机状态不可恢复，不能把一次回报差异当成稳定动作差异。

在 B 的可靠 SFT 起点完成信号检查后，才执行正式 C 训练；此前允许 1 步工程 smoke 检查梯度、检查点与日志。若所有候选同成同败，不启动 25 步 C 作业。信号关卡按完全相同的动作 token 合并重复候选，要求不同动作间有回报差异，且 M=8 两半续写得到的局部优势同号；这仅为小样本机制检查，不构成显著性结论。

## 验证与运行

- CPU 检查：首次终止、状态隔离、学生前缀/用户/工具的 loss mask、候选 0 token 一致性、局部损失归一化及一次 optimizer update。
- GPU smoke：复用已通过的普通 GRPO 环境检查；对新增 SFT 与局部分支训练链路分别做必要的单卡验证。
- B/C 短程：同一 SFT adapter，各 25 步，4任务×8条/步；同一学习率、seed和评测协议。
- 报告：train/test Pass@1、Pass⁴、Pass@4、逐任务成功数、写操作任务覆盖、工具错误、教师/分支 API tokens、策略 tokens 和 GPU 时间。
- test 仅在方案冻结后评测。短程机制验收不根据 test 选配置。
- 所有计算通过 Slurm；运行前冻结源码，输出不覆盖历史目录。共享 Slurm 壳只负责环境、资源和脚本调用。

## 实际入口与依赖

共用入口为 `scripts/train/grpo/run_verified_comparison.sh`，只接收阶段参数；底层采集、SFT、GRPO、评测各复用一个入口。通过 `COMPARISON_ROOT` 指定产物目录，`STUDENT_ADAPTER_PATH` 指定已修复 GRPO 的 step200 adapter。

| 阶段 | 固定预算 | 通过条件及下游 |
| --- | --- | --- |
| CPU 检查 | 基线回归、新增 mask/终止/分支/梯度测试、配置与 Ruff | 成功后允许 GPU smoke |
| `smoke` | 教师 12 条、学生纠错尝试 12 条、SFT 1 epoch、B 1 步 | 写任务成功、真实纠错成功、至少 2 条 SFT 数据，且 adapter 可加载 |
| `sft` | 教师 160 条、纠错尝试 80 条、SFT 2 epochs | 至少 4 条合格数据；SFT 完成后以固定协议各评测 train/test |
| `b` | 25 步、每步 4×8 条；step25 train/test 各 8 次/任务 | 汇总通过；保留单独 SFT 与 B 结果 |
| `c` | 1 步 smoke；40 个 train 根轨迹中仅指定 6 任务允许分支，至多 192 条 K4M8 后缀；关卡通过后 25 步 K4M2 | 无稳定局部信号则写 `c/stopped.json`，不启动 25 步；通过后同协议评测 |

各 GPU 阶段串行依赖，避免多个独立进程各自使用完整 MiMo 限额。正式 B/C 都从正式 SFT adapter 重新初始化，smoke 的权重不进入正式比较。SFT 保存的是可加载 PEFT adapter；其短程 optimizer/extra 文件不是完整恢复检查点。

产物位于 `experiments/verified_sft_grpo/minimal-seed42-v1/`，作业编号与冻结源码见其 `submission.json`。报告复用 `scripts/eval/summarize_qwen3_suite.py`，manifest 指定 `train_step=25/200` 时读取该检查点独立评测，不使用训练轨迹的平均奖励。

2026-10-20 提交记录：

| 作业 | 阶段 | 本次交付时状态 |
| --- | --- | --- |
| 215803 | 新 GRPO step200 test 补评 | RUNNING |
| 215827 | 最终 CPU 验收 | COMPLETED，exit 0 |
| 215829 | 采集/SFT/B smoke | PENDING，等待 A 补评释放 API 并发 |
| 215830 | 正式数据、SFT 及 SFT 评测 | PENDING，afterok:215829 |
| 215831 | B 25 步及独立评测 | PENDING，afterok:215830 |
| 215832 | C smoke、信号关卡、条件训练及评测 | PENDING，afterok:215831 |

GPU 源码固定为 `experiments/verified_sft_grpo/source-minimal-v3`。CPU 验收为 119 项基线回归 + 37 项新增/相关检查，真实 Qwen3 tokenizer 的教师正文/EOS 边界、B/C 训练配置一致性、7 个框架文件的 Ruff 检查均通过。早期 CPU 作业 215818 暴露并修复 object-array 转 tensor 与测试夹具问题；215823 的 156 项测试通过后修正 Ruff 导入/排版，历史快照和日志保留。

本次仅完成实现、CPU 验收和 GPU 排队安排；GPU smoke、正式训练及提升幅度尚未验收。后续阶段使用 `afterok` 与 `--kill-on-invalid-dep=yes`，前序失败时下游不继续消耗 GPU。没有自动提交 200 步扩展实验。

## 资源核对

2026-10-20 检查时账户为 msc，默认 QoS msc，可使用最多 2 张 Pro6000；无活跃作业。SSD 使用 76.7/150 GB，HDD 使用 188.2/400 GB。本轮优先单卡串行运行，避免教师与用户模拟器并发作业争用 100 RPM 配额，不使用可抢占 QoS。

官方 AI agent 摘要的 GitHub 页面和 raw 地址均返回 404，沿用仓库集群手册并核对实时 Slurm 限额。源码未提交作业前可编辑，历史快照保持不变。


## 2026-10-21 Flash 恢复实验

旧链路的实际结果：215803 已完成，修复环境后的普通 GRPO step200 在旧 Pro 用户模拟器下 train 为 104/320（32.5%），test 为 8/80（10%）。215829 在教师采集阶段因返回多个工具调用而失败，尚未完成轨迹、SFT 或 B 训练；215830–215832 随依赖失败自动取消。失败时教师模型为 Pro，不能将该失败归因于 Flash。

217264 的 Flash 连通性验证通过，但只检查普通文本响应。本轮保留严格单工具校验，为教师 API 请求副本增加单动作约束和有上限的纠错反馈，并在拒绝前记录实际调用数量。教师私有格式指导不写入学生历史；被拒绝的调用不执行，不产生虚假工具结果，也不作为监督目标。另修复小样本采集元数据：6 个任务的采集应按这 6 个任务验收，不能沿用完整 40 个训练任务。

新源码冻结于 `experiments/verified_sft_grpo/source-flash-minimal-v1`，输出位于 `experiments/verified_sft_grpo/minimal-flash-seed42-v1`。实际作业及检查状态以 `submission.json` 和 Slurm 记录为准。CPU 回归增加采集子集测试，并增加三种真实 Flash native tools 探针（单工具、独立工具需求、依赖工具需求），每轮只接受一个工具调用。

执行顺序：CPU/真实 API 预检 → 采集/SFT/一步 GRPO smoke → A 的 Flash train/test 补评 → 正式采集与 SFT/独立评测 → B 25 步与评测 → C smoke/局部信号检查/条件执行 25 步与评测。所有 GPU 阶段串行，前序失败则取消后续依赖，不自动扩展到 200 步。A 使用既有 Pro 训练的 step200 adapter，仅在 Flash 下重新评测，仍属于不同训练预算和训练模拟器的参考结果；B/C 共享 Flash 协议、SFT adapter、优化器设置及 25 步预算。

本轮由 explorer 调查失败日志与验证覆盖，fallback worker 修复教师 API 和探针，主线程修复子集元数据、集成检查并提交作业。2026-10-21 资源核对：正常 msc QoS，最多 5 个在队列中的作业、2 个运行作业；SSD 77.5/150 GB，HDD 188.2/400 GB。继续使用单卡 Pro6000 串行实验。


本轮预检 217534 已 COMPLETED（exit 0）：120 项基线回归 + 38 项新增/相关测试通过，配置、真实 Qwen3 tokenizer 和 7 个框架文件 Ruff 通过。三个真实 Flash 工具场景均一次请求通过。

| 作业 | 阶段 | 2026-10-21 提交时状态 |
| --- | --- | --- |
| 217534 | CPU + Flash tools 预检 | COMPLETED |
| 217536 | 采集/SFT/B 一步 smoke | PENDING |
| 217537 | A step200 Flash train/test 补评 | 等待 smoke 成功 |
| 217538 | 正式采集、SFT 及独立评测 | 等待 A 补评成功 |
| 217539 | B 25 步及独立评测 | 等待 SFT 成功 |
| 217541 | C smoke、信号关卡及条件训练/评测 | 等待 B 成功 |

提交记录不表示 GPU 实验已完成；尚无本轮 B/C 准确率或提升结论。
