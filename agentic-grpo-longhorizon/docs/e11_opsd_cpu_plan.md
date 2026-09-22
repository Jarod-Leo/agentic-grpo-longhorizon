# E11：纯 Agent-OPSD-FB 实施与 CPU 验收

前置验收：E06作业166809（127项）和E07作业166814（129项）均通过CPU测试。本轮依据用户最新要求恢复首版实验计划中的E11/E12实现，范围仅为代码和CPU测试，不启动GPU实验，也不声称已验证教师质量或训练收益。

## 方法与最小改动

E11从官方Qwen3-8B及相同LoRA初始化独立开始。复用E06的sampled-token clipped PG损失和评分钩子，RL系数0，自蒸馏系数1。

$$
A_{S,t}=\operatorname{stopgrad}(\log\pi_{\theta_{old}}(y_t\mid x,F_2,y_{<t})-\log\pi_{old}(y_t\mid x,y_{<t})).
$$

自教师是本批次更新前的当前actor（包含LoRA）。在任何optimizer更新之前同步完成整批评分并detach缓存；逻辑冻结，无需复制第二份模型。每个新批次重新评分，复用veRL compute_log_prob与FSDP加载/卸载机制。该资源优化等价于相同权重快照，不表示教师前向免费。

当前缺口只有环境反馈提取和自教师条件评分：在环境清理前提取反馈，经agent loop现有extra_fields进入批次；学生输入完全保持原样，仅教师的原始prompt前加入清楚标记的训练期反馈，原始response token IDs与assistant mask不变。反馈不包含原始参数、实体ID、隐藏答案或未来对话全文。

F2仅报告可核验的工具错误、完全相同的失败调用重复及终局状态，并给出不含答案的修复方向；不能凭失败推断遗漏确认、无来源实体或具体流程违规。环境污染时拒绝产生有效诊断。保存反馈规则版本、覆盖率与本批次policy_version。F0无反馈必须保持完整输入一致，可作数值负对照。

反馈token上限512，总教师输入上限24576；超过上限明确失败，不截断学生历史或答案。纯OPSD仍保留原终局指标，训练成功必须来自有效蒸馏信号，不能把奖励混合组当成自蒸馏有效性证明。

复用200步、4任务×8轨迹、40/10 train/test、50步归档HDD、100步评测train配置。沿用32768动态微批token预算，教师模型不调用32B服务。上述仅为未来运行默认值，本轮不提交训练。

## CPU验收与分工

worker实现可核验反馈模块、环境接线和对应测试；另一worker实现自教师批次构造、冻结评分及对应测试；主线程负责方案、共享配置/运行入口、验收和Git同步，explorer独立审查。

CPU检查覆盖：反馈依据与隐私字段隔离、环境清理前捕获、extra_fields传播、F0输入/评分一致、F2只改变教师上下文、精确token/mask对齐、拒绝超长、LoRA不被禁用、教师分数detach且本批次固定/下一批刷新、损失方向及系数退化。复跑E06/E07全部回归，只使用Slurm CPU节点。通过后再规划和实现E12。

方法依据：[项目首版计划](../../实验计划首版.md)的Agent-OPSD-FB定义；[OPSD论文](https://arxiv.org/abs/2601.18734)。F2是本项目的工具环境适配，不直接继承论文任务上的效果结论。

## 运行复用与边界

未来E11使用`run_qwen3_opsd.sh`→共用`run_qwen3_formal.sh`，集群入口复用`cluster_setup/qwen3_shared/run.sbatch`，只需一份actor权重。E06/E07仍使用已准备的teacher_pair.sbatch；不另复制实验运行壳。E11不访问32B评分端点，不改变MiMo角色。

日志中的`actor/opd_*`为共用sampled-token蒸馏损失字段；具体外教师/自教师由run.json的distillation.teacher.teacher_mode区分，不能仅按字段名判断方法。分支梯度范数是对采样token log-prob的导数，不是全模型参数梯度。CPU通过不代表FlashAttention2、去padding、动态微批的GPU执行、分布式通信、显存或速度已验证。

## 验收结果（2026-09-22）

CPU作业166829完成（1分17秒，退出码0），96项共用回归＋57项蒸馏/反馈测试全部通过。真实Qwen3 tokenizer构造的重复失败反馈样例为75 tokens（含边界标记），在512上限内；三组Hydra配置和7040条轨迹预算检查通过。

审查修复了F0空反馈到batch的接线、F0超长拒绝，并明确独立validation完全跳过训练反馈，避免改变二元评测。精确重复诊断使用保留大小写的parameters，未使用旧param_str小写近似。反馈及规则版本记录到轨迹日志，污染训练状态明确拒绝。

源码source-v1、原始日志和cpu_validation.json保存在experiments/e11_opsd/；Slurm实际分配4CPU/12GB/0GPU。未启动GPU作业，未验证GPU计算路径及教师实际指导价值。
