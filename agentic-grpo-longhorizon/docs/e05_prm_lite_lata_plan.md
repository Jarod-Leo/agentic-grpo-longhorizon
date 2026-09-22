# E05：GRPO + PRM-Lite + LATA 实施与执行计划

## 实验目标与冻结设置

按照根目录《实验计划.md》的 E05，复现原项目代表性联合方案，并与 E01 Vanilla GRPO 比较。用户于2026-09-22确认本分支也训练200步，每50步保存完整checkpoint至HDD，每100步在train集独立评测。四个训练分支中的E05从官方Qwen3-8B及seed42的相同LoRA初始化开始，不续训E01。

| 项目 | 冻结配置 |
| --- | --- |
| 模型与训练 | Qwen3-8B BF16，非思考，LoRA r16/alpha32/all-linear，seed42 |
| 数据 | `experiments/sft_collect_airline/split.json`的40 train/10 test；不新增dev |
| 每步采样 | 4任务×8轨迹；train temperature=1/top_p=1/top_k=-1 |
| 优化器 | AdamW lr=5e-6，constant；无额外KL、entropy、OPD |
| 性能 | FlashAttention2、remove-padding、dynamic microbatch、32768 token预算、seq-mean-token-mean |
| 长度与轮数 | 沿用E01的8192 prompt/12288 response、24576 model length及15轮限制 |
| PRM-Lite | 原v4规则；过程分数截断至[-0.5,0.5]，系数0.3 |
| LATA | 已注册的grpo_lata，alpha=1.05；保留原有效轨迹公式 |
| 总预算 | 200更新，20 epochs；不是旧配置的300步 |
| 独立评测 | 第100/200步，各40 train任务×8轨迹，temperature=.7/top_p=.9 |
| checkpoint | 第50/100/150/200步；先写SSD、完整归档HDD后保留SSD链接 |

训练6400条轨迹，加周期评测640条，共7040条。10个test任务留到方法与比较规则冻结后统一评测；不能用train过程奖励代替独立任务成功率。主指标Pass@1（任务宏平均成功率），可靠性Pass^4，补充Pass@4。

## 已有能力与确认的接入缺口

复用`src/envs/tau_bench_interaction.py`的PRM-Lite v4、工具记录与MiMo客户端，复用`verl/verl/trainer/ppo/core_algos.py`的grpo_lata；无需奖励模型或额外推理GPU。旧`run_exp4_prm_lite_lata.sh`依赖/workspace、旧SFT模型及本地用户模拟器，不能直接当作当前Qwen3实验入口。

当前`TauBenchAgentLoop`将优化奖励强制当成二元任务结果，PRM非二元奖励会触发断言。最小修复为：轨迹`score`始终记录二元outcome，另存`training_reward`、`process_score`和`reward_mode`。训练使用合成奖励，validation向veRL返回二元outcome；两者日志与汇总分开验收。现有extra_fields已经携带outcome/process字段，无需重建reward manager或训练循环。

模式由主Hydra配置`tau_bench_reward_mode`确定，公共准备脚本将其写入run.json并注入共用MiMo interaction配置。联合入口只选择配置并调用共用formal入口，避免复制运行逻辑或用户模拟器配置。

## 实际算法语义与数值修复边界

记二元结果为 $o_i$，原v4过程分数为 $p_i$，训练使用：

$$
R_i=o_i+0.3p_i,\qquad p_i\in[-0.5,0.5].
$$

LATA首先在同任务的8条轨迹内对 $R_i$ 做GRPO标准化，得到 $A_i$。当前实现按完整response中的绝对token位置 $t$ 加权；用户/工具token和padding通过mask排除。令有效assistant token数为 $L_i$：

$$
w_{i,t}=\frac{L_i\,\alpha^{L_i-1-t}}{\sum_{s:m_{i,s}=1}\alpha^{L_i-1-s}},\qquad
\widetilde A_{i,t}=A_i\,w_{i,t}\,m_{i,t}/\sqrt{L_i},\quad\alpha=1.05.
$$

这是原代码的**token位置加权并按长度进一步缩放**，并非按真实对话轮ID加权。当前损失聚合还会做每条轨迹token均值，不能将其解释为已抵消长度稀释，也不能预先宣称它保护长推理。为复现现有对照，本实验不擅自改为真实轮权重、不把除sqrt改成乘sqrt、不改变损失聚合方式。

已确认边界缺陷：全零response mask会产生NaN；稀疏且靠后才出现有效token时，masked位置指数值可能在float32转换后成为inf，再乘0产生NaN。只修复masked归一化的数值实现，对全零mask明确报错；有效轨迹的既定公式不变。报告标明是对原LATA实现的数值稳定化，而不是声称所有源代码均未变化。

## 实施清单

1. 增加`configs/train/grpo/qwen3_prm_lite_lata.yaml`和一个很薄的训练入口；复用Qwen3 common/performance与共用Slurm壳。
2. 修复共用Tau agent loop的奖励/结果分离；保留二元指标、完整分组、输入协议、思考关闭、API失败与checkpoint的严格验收。
3. 完成LATA等价数值稳定化及回归测试；PRM-Lite规则本身保持原样。
4. 共用run metadata/summary记录模式与实际训练奖励信号，分别统计outcome混合组、训练奖励非恒定组及outcome饱和但PRM提供差异的组。
5. 将现有PRM v4规则测试、联合loop测试、LATA CPU测试、配置可比性与汇总检查纳入同一个CPU Slurm预检。
6. 现有LoRA初始化指纹日志增加初始B张量的nonzero/numel实测计数，不改变初始化本身。冻结新源码后做单卡真实1步验证，检查32条轨迹、有限奖励/优势/梯度、初始B确为零且更新后B非零，以及完整checkpoint和HDD归档。通过后自动启动全新初始化的200步正式作业。

所有活跃E01作业继续使用其source-formal-v4快照；本次更改只进入新E05快照。历史失败、验证和正式结果均保留。

## 资源、MiMo额度与提交顺序

2026-09-22实时检查：msc允许同时2块PRO6000、2个运行作业；当前E01作业165407占1块。因此可以提交另一项独立单PRO6000实验；不改成双卡训练。

但是两个作业使用同一个MiMo key，额度合计仅100RPM/1000万TPM。当前E01快照内的限流器是进程内独占配置，不能在保持其运行代码冻结的同时让两个客户端各按100RPM请求。新实验的CPU验证立即推进；真实MiMo smoke设为`afterany:165407`，正式训练设为`afterok:<smoke>`。这明确意味着新作业会申请自己的单卡资源，但不同时运行两组MiMo采样，也不提前占着GPU等待接口额度。若后来提供另一个独立额度的凭据入口，可以另行安排并发，不能推断同key有额外额度。

资源申请沿用已验证的单PRO6000、36小时正式上限；真实smoke申请30分钟。HDD模型按已批准的大文件顺序加载方式使用，固定已验证可访问模型的gpu-pro6000-11。复用CPU检查壳及GPU run.sbatch，不新建一套cluster逻辑。

存储检查：SSD约126.8/150GB，HDD约141.6/250GB。预计剩余E01 checkpoint约17.1GB，E05单步验证约17.1GB，E05正式四份约68.4GB，总量约244.2GB，可容纳但余量有限。先后执行避免两个作业同时在SSD写约17GB的checkpoint；不删除旧checkpoint。开跑前复查真实余量。

200步预算优先于原E05的20/30/50步候选；全矩阵原80GPU小时预算需依据新单步实测更新，不能据此宣称仍满足原总预算。正式时间估计以本分支真实smoke为准，不直接套用E01后期缩短的对话耗时。

## 验收与交付

CPU：验证PRM规则；真实奖励字段沿异步loop/batch传递；二元指标不被shaped reward污染；LATA不同长度/稀疏mask/零mask/相同奖励组/PRM饱和组梯度；配置仅有方法差异；数据划分和7040条预期数量一致。

GPU smoke：完整一批32轨迹，使用真实MiMo与相同Qwen3输入协议；有限优势及actor梯度，adapter权重有限且B非零，checkpoint含模型/优化器/状态/数据游标，SSD链接目标在HDD。任何验收失败阻止正式作业启动，不以删除异常轨迹或放松验收掩盖问题。

交付物：本计划、冻结配置和源码manifest、提交记录、单步验收与耗时、200步作业编号、HDD checkpoint路径。正式训练完成后生成统一train结果；test评测按最终矩阵协议执行。代码与小型报告同步用户指定GitHub仓库，凭据、模型、原始大日志不提交。

本次实际多agent分工：explorer调查原方法和接入缺口；两个worker分别实施算法/loop及配置/共用运行汇总；主线程制定决策、写计划、执行Slurm验证、审查结果并提交新作业。

## 2026-09-22 提交记录

冻结源码：`experiments/e05_prm_lite_lata/source-v1`，576个文件的SHA-256已逐一核验；清单及小型验收记录保存在同级实验目录。

| 作业 | 编号 | 提交时状态与验收 |
| --- | --- | --- |
| CPU预检 | 166634 | COMPLETED，39秒，94项测试通过；配置可比性、tokenizer与7040条预期轨迹均通过 |
| 单卡真实一步验证 | 166639 | PENDING，`afterany:165407`；30分钟上限，尚未执行GPU验证 |
| 单卡200步正式训练 | 166640 | PENDING，`afterok:166639`；36小时上限，验证成功才启动 |

提交核验时E01的165407仍在运行，已完成184步。两个E05作业均申请1块PRO6000，尚未分配GPU；依赖未满足时不占用GPU。验证失败会取消依赖它的正式作业。

正式checkpoint归档位置：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e05_prm_lite_lata/formal-seed42-200-v1/`。预期生成`global_step_50`、`global_step_100`、`global_step_150`及`global_step_200`。验证checkpoint单独归档到`smoke-seed42-v1/`。

状态命令：`squeue --me`；查看本次三个作业：`sacct -j 166634,166639,166640 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList`。本轮停止持续监控；实际一步耗时与更新验收将在验证运行后写入`smoke-seed42-v1/train/readiness.json`。

Git同步状态：代码与小型验证记录已提交本地`e05-prm-lite-lata`分支。GitHub推送因当前VS Code Git认证会话失效而失败，尚未上传；恢复Git登录后可推送该分支。此问题不影响已提交的Slurm作业。
