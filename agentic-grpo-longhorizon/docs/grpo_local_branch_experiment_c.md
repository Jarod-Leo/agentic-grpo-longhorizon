# 实验 C：GRPO + 局部分支优势

## 目标与边界

研究同一中间状态下的候选动作比较，是否改善长链路工具调用的信用分配。
本轮授权开展验证实验，不启动从 base 开始的 200 次更新正式训练。
初始诊断冻结 E01 step 200，仅使用原有 40 个 train 任务；10 个 test 不参与方法选择。

## 模型和协议

- Base：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-8B`。
- E01 checkpoint：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3/global_step_200`。
- 冻结诊断加载上述 checkpoint 的 `actor/lora_adapter`，不加载优化器、不执行梯度更新、不写完整 checkpoint。
- 新验证使用 MiMo `mimo-v2.6-pro`，100 RPM、1000 万 TPM。历史 E01/B 使用 v2.5，历史数字不能作为新协议下的严格配对对照。
- 成功判定沿用原始 tau-bench 二元结果，不增加 LLM judge、不更改奖励规则。
- 候选及续写使用同一冻结模型；训练分布诊断采用 temperature=1、top_p=1，区别于历史 test 的 0.7/0.9。

## 分支方案

在 assistant 生成前保存完整状态。每个分支点采样 K=4 个候选 assistant turn；
每个候选只生成一次，固定其 token 和工具调用，再从执行该动作之前的相同状态独立续写 M=2 次。
分支保留已执行工具造成的数据库变更、用户历史、动作历史及已用轮数/token 预算，不能重新获取完整预算。
首版动作粒度为完整 assistant turn，不声称能区分同一轮内多个工具调用的贡献。
不同候选允许相同文本，不通过拒绝采样强制多样性。

$$
\widehat Q_{t,k}=\frac{1}{M}\sum_{m=1}^{M}R_{t,k,m}
$$

$$
A^{\mathrm{local}}_{t,k}=\widehat Q_{t,k}-\frac{1}{K-1}\sum_{j\ne k}\widehat Q_{t,j}
$$

后续训练拟采用 $L=L_{\mathrm{GRPO}}+\lambda L_{\mathrm{local}}$，初始候选权重为 0.25。
局部项只覆盖候选动作 token；前缀及后续续写不参与该项损失。
冻结诊断只计算候选回报与局部优势，不能替代梯度 mask/优化器集成验收。

## 验证顺序与门槛

1. CPU：验证数据库/历史/状态隔离，候选固定后独立续写，轮数与预算继承，失败传播和资源清理。
2. GPU smoke：单卡 Pro6000，少量 train 分支点；核对 K×M 完整性、二元奖励、执行成本和日志。
3. 固定模型信号诊断：计划 80–120 个 train 分支点。记录有效分支率（候选平均回报不全相同）、候选重复率、同候选续写差异、tokens/API/时间。
4. 对部分分支点额外续写确认动作排序稳定性。有效分支率约 20% 仅是预先提出的筛选门槛，不是统计显著性标准；M=2 的差异不足以证明有效信用信号。
5. 信号及成本通过后，才接入局部训练项并从同一 E01 完整 checkpoint 做 20–30 次更新的 C/普通 GRPO 对照。恢复方式、优化器状态、用户模拟器一致。

任何阶段只运行完成，都不等于后续阶段通过。若所有候选同成同败，局部优势为零；若续写随机性大于动作差异，应调整分支位置或 M，再考虑训练。
短程训练验收关注有限 loss/梯度、目标 mask、KL/长度/工具行为和无明显退化，不要求短期 train 成功率显著上升。

## 正式实验（尚未启动）

通过上述验证后，先做原始 Qwen3-8B 短 smoke，再进行 base 同起点的普通 GRPO 和 C 各 200 次实际更新。
每 50 次更新保存，每 100 次更新评测 train，最终用同一协议评测 test。
保持根轨迹 4 任务×8 条；额外分支成本单独报告。补充相近交互预算比较与多个种子；
需要区分机制时，增加相同分支设计/目标 mask、但采用任务级基线的消融。
测试不做分支搜索，报告 Pass@1、Pass@4、Pass⁴，以及逐任务结果和交互成本。

## 实现原则

复用当前 veRL 服务、共享评测入口、TauBenchAgentLoop 工具执行与奖励计算。
新增可复用快照/恢复及诊断子类，避免复制整套训练框架；快照不得深拷贝网络客户端/锁。
主轨迹和分支分别标识，但 API 账单保留全部真实调用。
CPU 及 GPU 工作通过 Slurm 执行，提交前冻结源码，运行期间不改快照。

## 运行记录

2026-10-19：开始实现。资源检查：msc QoS 允许 2 张 Pro6000；SSD 约剩 82 GB，HDD 约剩 26 GB。
官方 `NTUEEECluster/docs/main/agent.md` 与 GitHub 页面均返回 404，本轮沿用仓库集群手册并核对实时 Slurm 规则。
CPU 首轮 211995：29 项测试通过，随后配置检查发现 local_branch 嵌套对象类型错误，已补 Hydra target。
CPU 第二轮 212002：31 项测试及配置检查通过；快照未包含 veRL 的 pyproject.toml，Ruff 误用外层规则导致退出，已修复共享快照入口。
CPU 最终验收作业 212004 使用 source-v3，31 项测试、Hydra 配置解析和 Ruff check/format 全部通过，exit code 0。
GPU smoke 作业 212006 已提交，单卡 Pro6000、最长 2 小时；信号诊断 212007 依赖 `afterok:212006`，单卡、最长 6 小时。
后者只在 smoke 的执行与 branch_summary 工程验收均成功后启动，依赖失败自动取消。提交时两个 GPU 作业仍排队，尚无实验指标。
GPU smoke 212006 使用 v2.6 通用别名收到 HTTP 400，未开始模型生成、未产生分支。官方模型清单确认 API ID 需使用 `mimo-v2.6-pro`、`mimo-v2.6-flash` 或 `mimo-v2.6-pro-ultraspeed`；本实验已改为旗舰 `mimo-v2.6-pro`。
212007 因依赖失败自动取消。CPU 检查已加入对具体模型 ID 的断言；修复版本等待重新验收。尚未提交短程参数更新或 200 次更新正式训练。

诊断实现另外禁止 test split，并跳过已终止或受污染的前缀。候选和续写均经过原有生成审计。
MiMo 进度的终止轨迹总数因跳过分支而未知，日志显示 `N/?`；主轨迹与实际分支数量以 branch_summary.json 分别统计。

## 入口与产物

- CPU：`scripts/train/grpo/check_local_branches.sh`，经 `cluster_setup/qwen3_shared/check.sbatch` 运行。
- GPU：`scripts/eval/eval_local_branches.sh`，经 `cluster_setup/qwen3_shared/run.sbatch` 运行。
- Smoke：`BRANCH_STAGE=smoke SAMPLES=1`，40 条 train 主轨迹，仅 task 0/1/2/4 在第 2 个 assistant turn 前分支。
- Signal：`BRANCH_STAGE=signal SAMPLES=3 BRANCH_ANCHOR_TURNS=[2,4,6]`，每个任务的三条主轨迹分别选择一个锚点，最多 120 组；未到达或不合格锚点会记录 skip。
- 目录：`experiments/grpo_local_branch/`。分支逐组数据：`<run>/eval/branches/worker-*.jsonl`；汇总：`<run>/eval/branch_summary.json`。
- 原始主轨迹、工具审计和 API 日志分别为 `eval_trajectories.jsonl`、`tool_audit.jsonl`、`api.jsonl`。

branch_summary 的 `engineering_accepted` 是执行完整性验收；`signal_screen` 是描述性筛选，不能代替额外续写稳定性验证或短程训练验收。

## 2026-10-19：终止处理复核与修复后重验

source-v4 的 smoke 212144 完成，用时 20 分 46 秒；signal 212145 完成，用时 1 小时 3 分 44 秒。
signal 的 120 条主轨迹产生 42 个实际分支点、168 个候选、336 条后缀续写；原始候选样本均值有差异的分支点为 4/42（9.52%）。
其中两个分支点的四个候选 token 完全相同，均值差异来自续写随机性，不能视作不同动作的信用信号。

虽然旧汇总的已有工程检查全部通过，后续复核发现共用终止处理有缺陷：
`transfer_to_human_agents` 在 tau-bench 中已经结束环境，但 veRL 工具处理后仍返回生成状态。
107 条主轨迹与 243 条分支续写在成功转人工后继续生成；78 个跳过记录全部发生在锚点前已成功转人工的轨迹。
上游环境在终止评分时会重建目标数据库，因此终止后的继续执行可能污染状态或重复累计奖励。
旧运行保留用于故障追溯，不能作为修复后协议的性能或信用信号结论；历史模型指标也需复评，尚不能量化影响。

修复范围保持在现有共用入口：

1. tau-bench 适配层在首次环境终止后停止生成、工具执行和用户模拟器请求，保留首次终止结果，不改变原始成功判定。
2. C 的候选及后缀统一复用修复后的 TauBenchAgentLoop；跳过原因区分锚点前终止、预算不足、未到达锚点及污染前缀。
3. 共用评测和 C 汇总增加终止后调用检查。新运行标注 `terminal_semantics=first_environment_done_v1`，不覆盖历史产物。
4. 保留原始候选样本和原始优势用于追溯；诊断按完全相同的候选 token 合并续写统计，将原始样本差异与动作间差异分别报告。
5. 新 smoke 仍使用 K=4、M=2；信号诊断保持 40 个 train 任务×3 条主轨迹、锚点 [2,4,6]，将 M 增加到 8，报告独立两半续写的动作排序一致性。该指标仍是描述性诊断。
6. 使用同一修复源码和 MiMo v2.6 Pro，通过共享 `eval_qwen3.sh` 重新评测 E01 step200：train 40×8、test 10×8，temperature=0.7、top_p=0.9。

提交顺序为 CPU 回归检查 → GPU smoke → C 信号诊断 → E01 train 复评 → E01 test 复评，均用 `afterok` 依赖串行执行，避免多个作业同时争用 100 RPM 的 MiMo 配额。
只申请单卡 Pro6000；C 信号诊断上限 6 小时，其他 GPU 验证上限 2 小时；不写完整 checkpoint、不进行参数更新。
源码在提交前冻结，提交编号和输出路径记录在各运行目录的 `submission.json`。

修复快照为 `experiments/grpo_local_branch/source-v5`。CPU 作业 212674 已完成，70 项 pytest、Hydra M8 配置、shell 语法及既有 Ruff 检查全部通过；历史日志回归检查成功拒绝 source-v4 中终止后的继续执行。

| 阶段 | Job ID | 前置作业 | 运行目录（相对 experiments/） |
| --- | --- | --- | --- |
| CPU | 212674 | — | `grpo_local_branch/cpu-check-v6` |
| GPU smoke，K4/M2 | 212684 | 212674 | `grpo_local_branch/smoke-e01-step200-terminal-v1` |
| C 信号诊断，K4/M8 | 212685 | 212684 | `grpo_local_branch/signal-e01-step200-terminal-m8-v1` |
| E01 train，40×8 | 212686 | 212685 | `e01_vanilla_grpo/train-step200-terminal-v1` |
| E01 test，10×8 | 212687 | 212686 | `e01_vanilla_grpo/test-step200-terminal-v1` |

提交时上述 GPU 作业尚未完成。C 的 pooled 信号未达阈值不等于工程失败，依赖链只要求执行及工程检查成功，不会自动进入参数更新。

### 修复后 GPU 结果

212684 smoke 已完成，用时 17 分 57 秒，4 个分支点、32 条后缀续写，工程验收通过。
212685 信号诊断已完成，用时 56 分 39 秒，工程验收通过，结果如下：

- 120 条主轨迹中，44 个分支点执行完成；76 个在锚点前已经终止，正确记录为 `terminal_before_anchor`。
- 176 个候选动作，每个动作 8 次后缀续写，共 1,408 条后缀；其中 352 条成功（25%）。这是条件分支回报，不能作为新模型的独立评测指标。
- 原始候选均值有差异的分支点为 0/44；合并相同 token 动作后的动作差异同样为 0/44。全部局部优势为零。
- 36 个分支点含至少两个不同的 token 动作；前后各 4 次续写均未形成可比较的动作排序。
- smoke 与 signal 的终止后生成、工具执行、用户 API 请求均为 0。signal 的 1,128 条发生成功转人工的轨迹均通过终止审计。
- signal 共 1,442 次 MiMo 请求，全部成功；API 总 token 为 1,099,225，其中分支消耗 1,017,403。

结论：终止处理修复已通过本次 CPU/GPU 验证，但当前 E01 step200、固定锚点 [2,4,6]、K4/M8 的方案没有获得非零局部信用信号，未通过进入短程参数更新的筛选。不能将工程运行成功视作算法验证通过。
E01 train 复评 212686 已启动，test 212687 等待其成功结束；最终独立成功率以各自 `eval/summary.json` 为准。
