# E07：GRPO＋OPD 实施与 CPU 验收

E06 已由 CPU 作业166809通过127项测试和真实tokenizer/配置检查后，开始本组实现。E07从相同官方Qwen3-8B与LoRA初始化独立开始，不续训E06。本轮只实现与CPU验证，不执行GPU测试或训练。

## 最小改动

把E06中已验证的教师与训练日程抽为`qwen3_distillation_common.yaml`，两组共用教师、评分接口、actor forward、token mask和汇总，仅由配置选择损失系数。E07采用：

$$
L=L_{GRPO}+0.3L_{OPD}.
$$

GRPO维持二元终局奖励和原组内标准化。OPD保持E06的固定sampled-token teacher-old差、PPO clipping与assistant mask；不对OPD做GRPO组标准化，不以终局成败筛掉轨迹。两个分支使用同一次actor forward，各自记录loss及对sampled-token log-prob的系数加权梯度范数；后者不是模型参数梯度范数。

默认仍为200步、40/10任务划分、4×8 rollout、50步保存HDD、100步评测train。未来启动复用E06的同节点双GPU壳，仅设`QWEN3_CONFIG=qwen3_grpo_opd`；本轮不提交该壳。

## CPU验收

联合参数更新必须符合独立GRPO与0.3倍OPD梯度相加；冲突方向的信号能体现系数；GRPO饱和组仍保留OPD贡献。蒸馏系数0时，不访问教师且严格退化原GRPO。复跑全部E06评分/冻结/mask/对齐、既有GRPO/PRM/LATA回归，并验证E06和E07的Hydra配置仅有两项系数差异。

只有E06/E07两组都通过CPU验收后，再开始E11/E12的实施规划。GPU显存、真实教师效果与训练收益仍未验证。

## 验收结果（2026-09-22）

CPU Slurm 作业166814完成（52秒，退出码0），95项共用回归＋34项蒸馏测试全部通过。真实Qwen3-8B/32B tokenizer一致，两组配置仅有目标系数差异，200步输入预算均为7040条轨迹。原始日志、不可变source-v1与机器可读cpu_validation.json保存在experiments/e07_grpo_opd/。未提交GPU作业。
