# E06：纯 OPD 实施与 CPU 验收

用户于 2026-09-22 要求按 E06 → E07 → E11 → E12 顺序实现；每组通过 CPU 测试再推进下一组，本轮不提交 GPU 测试或训练。本文先冻结 E06，后续各组另记配置差异和验收证据。

## 复用范围与缺口

复用当前 veRL 的学生 rollout、真实 token 序列、assistant response mask、LoRA、动态微批、优化器、checkpoint 和 tau-bench 二元评测。当前副本缺少新版 veRL 的蒸馏模块，不能仅切换一个现成 OPD 配置；LoRA reference 快捷路径会禁用学生 adapter，也不能代表独立 Qwen3-32B 教师。

最小新增为：独立 sampled-token 蒸馏损失、更新前同步教师评分接口、基于 Transformers 因果前向的冻结教师服务。模型与服务仍使用现有依赖，不升级框架。教师接收学生原始 token IDs，不生成替代轨迹，不重新编码学生输出。

## 方法与配置

E06 从官方 Qwen3-8B 和 seed42 的相同 LoRA 初始化开始。外部教师为冻结的 Qwen3-32B BF16，revision 为 `9216db5781bf21249d130ec9da846c4624c16137`。教师与学生核对 token→ID 词表、特殊 token 与 chat template；模型输出词表的额外 padding 行不等于 tokenizer 不兼容。

对于本批学生采样的 token，固定旧策略与教师评分：

$$
A^D_t=\operatorname{stopgrad}[\log\pi_T(y_t\mid h_t)-\log\pi_{old}(y_t\mid h_t)],\qquad
\rho_t=\exp(\log\pi_\theta(y_t\mid h_t)-\log\pi_{old}(y_t\mid h_t)).
$$

$$
L_D=-\operatorname{Agg}_{m_t=1}\min[\rho_t A^D_t,\operatorname{clip}(\rho_t,1-\epsilon_l,1+\epsilon_h)A^D_t].
$$

沿用 actor 的 clipping 和 `seq-mean-token-mean` 聚合；只优化 assistant 文本和工具调用，用户消息、工具返回及 padding 不贡献损失。teacher、old 和蒸馏 advantage 均 detach。E06 的 RL 系数为 0、蒸馏系数为 1，完全跳过任务奖励 PG；环境仍产生二元结果用于诊断。全失败组不丢弃，蒸馏信号不作 GRPO 组内标准化。这是 clipped sampled-token PG 蒸馏，不是精确全词表 KL。

每批全部 rollout 完成后，冻结该批版本的 old log-prob，完成全部教师评分，才允许 actor 更新。服务核验 tokenizer 指纹、输入 token 哈希、教师 ID/revision、policy version 与 response 位置；任意不一致中止该批。教师采用严格因果 shift，不看到目标 token 之后的消息。

配置预设沿用 200 步、每 50 步归档 HDD、每 100 步评测 train，40 train/10 test、4×8 轨迹、32768 token 动态微批和现行非思考采样。此处只是保持可比的配置默认值，本轮不执行 GPU 作业。外部教师未来与学生各占同节点一块 GPU、同一 Slurm step；本轮只准备入口并测试协议。

## CPU 验收

- 小型随机因果模型验证 teacher/student 同权重同输入评分一致；偏移一位、mask 或指纹不匹配必须被发现。
- 改动未来 token 不影响更早位置；teacher 无梯度，学生梯度方向正确，正负信号和 clipping 均正确。
- 纯 OPD 不受环境 advantage 变化影响；全失败组仍可产生蒸馏梯度；用户/工具/padding 无梯度。
- 真实 HTTP 回环验证逐 token 评分协议；batch padding 压缩与回填保持顺序。
- CPU actor 微批实际更新验证接入；原 GRPO/PRM-Lite/LATA 测试保持通过。
- 配置、metadata 和汇总区分任务成功率与蒸馏信号。测试不调用 MiMo，不下载模型，不加载 8B/32B 权重。

CPU 测试全部经 CPU Slurm 作业执行，冻结独立源码快照。GPU 显存、FlashAttention 数值一致性、真实教师能力、吞吐和任务提升均保留为后续未验证事项。

方法参考：[veRL OPD 官方文档](https://verl.readthedocs.io/en/latest/algo/opd.html)。本项目保留当前 vendored 框架，通过最小接口实现其 sampled-token PG 目标，不声称直接运行了最新上游整套蒸馏框架。

## 验收记录

首轮CPU作业166791（source-v1）在31秒后失败：94项通过，既有GRPO回归捕获disabled蒸馏分支对未初始化零损失变量的引用。已在部署源码修复，保留原冻结快照和失败日志；后续使用source-v2重测。

policy_version是传输一致性标签，不能单独证明旧策略身份；CPU actor测试直接验证两轮更新过程中使用的是同一批固定old_log_probs。当前bypass模式的old来自rollout记录，不是每个microbatch重新计算的current。

第二轮CPU作业166805通过全部95项共用回归与E01/E05配置检查，但新测试收集时遇到依赖包的`scripts`命名冲突。将测试改为按绝对路径加载已有服务入口，不改变生产教师评分逻辑；source-v3继续验收。

第三轮166806：95项共用回归及31项蒸馏测试通过；tiny Qwen3交叉评分触发已安装的CUDA-only FlashAttention交叉熵，CPU无法执行。该CPU测试改选veRL已有PyTorch回退，保留真实Qwen3和actor forward；不据此声称验证了GPU/FlashAttention。source-v4继续验收。

第四轮166807：95项回归与32项蒸馏测试全部通过；实际Hydra实例化发现新增嵌套配置缺少`_target_`、仍为dict。补齐DistillationConfig目标，使配置校验与训练实际实例化一致；source-v5完成最终验收。

最终CPU验收：作业 **166809**，source-v5，COMPLETED/0:0，57秒。95项共用回归＋32项蒸馏测试，共127项全部通过；实际8B/32B tokenizer及模板指纹一致，Hydra实例化和7040条轨迹metadata正确。未执行GPU测试或训练，未调用MiMo。小型真实Qwen3交叉评分使用CPU PyTorch路径，不覆盖FlashAttention/GPU精度。记录见`experiments/e06_opd/cpu_validation.json`。
