# 原始 Qwen3-32B：test 集评测

2026-09-26 用户要求评测 E06 使用的 Qwen3-32B 教师模型在 test 集上的任务表现。本实验直接让32B模型作为工具Agent交互，不是读取其对8B轨迹的token评分。

## 固定协议

| 项目 | 设置 |
| --- | --- |
| 模型 | 原始 `Qwen/Qwen3-32B`，BF16，无adapter，无训练 |
| 模型目录 | `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/models/Qwen/Qwen3-32B` |
| revision | `9216db5781bf21249d130ec9da846c4624c16137` |
| 数据 | 原40/10划分中的test10任务，ID：3、10、16、20、23、26、30、36、43、46 |
| 正式重复次数 | 每题8次，共80条轨迹 |
| seed | 42 |
| 采样 | temperature=0.7，top_p=0.9，top_k=-1 |
| thinking | 关闭，并用输出审计验收 |
| 用户模拟器 | MiMo v2.5，沿用共享交互配置与已有API限流 |
| 长度/轮数 | 沿用prompt8192、response12288、context24576、assistant/user各15轮限制 |
| 判定与指标 | 原始tau-bench二元成功判定；任务宏平均pass¹、pass⁴、pass@4 |
| 资源 | 单节点单张Pro6000；正式时限2小时 |
| 内存/并发调整 | `gpu_memory_utilization=0.9`，`max_num_seqs=4` |
| 纯推理模式 | `trainer.val_standalone=true`，`resume_mode=disable`，`load_format=auto`，`free_cache_engine=false` |

模型17份权重分片均已核对在位，合计65,524,328,560 bytes。HDD模型按已批准方式顺序加载；活动日志、缓存和数据写入项目SSD。

## 复用与提交

复用 `scripts/eval/eval_qwen3.sh`、`configs/eval/qwen3/eval_qwen3.yaml`、共享准备/汇总器和 `cluster_setup/qwen3_shared/run.sbatch`。Slurm壳增加`"$@"`参数透传，使已有入口能接收每次运行的Hydra覆盖值，并在环境激活后恢复提交时明确指定的`POLICY_MODEL`；没有复制一套32B评测代码。

首次两轮提交复用E06已验收的`experiments/e06_opd/source-gpu-v1`，602文件manifest逐一校验一致，manifest SHA256为 `db88b4ebbbd2620b7251e37768bae563085788b89142902394ba7edf91f7fd18`。当前独立推理修复冻结于`experiments/e06_opd/source-eval-standalone-v1`，604文件manifest SHA256为 `4b893051f1d360c7852ba10b8d8785a6c8d898525ca4e57f44d8f023f478d041`。成功判定`tau-bench/tau_bench/envs/base.py`与E00/E01 test冻结版本相同，SHA256为 `20f77ff9255ecaa7db0fbd0972a81881393f282766992c94d572e442cddb8a39`。方法配置新增显式binary字段及环境变量传递，实际仍为binary。

| 作业 | 用途 | 输出目录 | 提交状态 |
| --- | --- | --- | --- |
| 171797 | 首次短验证提交 | `experiments/e06_opd/teacher-qwen3-32b-test-smoke-seed42-v1/` | 已取消：激活脚本覆盖模型为8B，未产生轨迹 |
| 171798 | 首次正式提交 | `experiments/e06_opd/teacher-qwen3-32b-test-seed42-v1/` | 未启动即取消，避免继承错误模型 |
| 171801 | hybrid模式的32B短验证 | `experiments/e06_opd/teacher-qwen3-32b-test-smoke-seed42-v2/` | FAILED：FSDP模型占用GPU，vLLM可用显存不足，无轨迹 |
| 171802 | 第二次正式提交 | `experiments/e06_opd/teacher-qwen3-32b-test-seed42-v2/` | 依赖失败自动取消，未启动 |
| 171809 | 独立推理，每题1次，10条轨迹 | `experiments/e06_opd/teacher-qwen3-32b-test-smoke-seed42-v3/` | 已提交，待验收 |
| 171810 | 独立推理，每题8次，80条轨迹 | `experiments/e06_opd/teacher-qwen3-32b-test-seed42-v3/` | `afterok:171809`；依赖失败自动取消 |

首次启动的`setup/activate.sh`无条件设置8B的`POLICY_MODEL`，覆盖了sbatch继承的32B路径。通过实际模型参数量8.19B和run.json发现后立即取消，模型尚在初始化，未生成任何评测轨迹。修复位于共用Slurm壳，默认8B行为保留，显式指定模型时尊重提交参数；重提使用新输出目录并记录替换关系。

第二次短验证171801正确加载了32.76B模型，但veRL原`val_only`仍初始化FSDP训练模型。vLLM启动时报实际可用32.39/94.97 GiB，低于0.9预算的85.47 GiB，8分41秒后失败。这是实际显存冲突，不靠降低到连模型权重都容纳不下的预算规避。

最小通用扩展位于`RayPPOTrainer.init_workers`：显式启用`trainer.val_standalone`时复用已有`AgentLoopManager(config, worker_group=None)`，跳过训练actor和训练资源池；`_load_checkpoint`在该模式中不访问actor。仍调用同一`fit → _validate`、工具交互、reward、输出与汇总器。仅支持无adapter原始模型的异步val-only、禁止恢复或额外critic/reference/neural reward；默认false不改变训练和checkpoint评测路径。未修改历史冻结源码。

CPU作业171807通过13项测试，包含实际fit的checkpoint-disable/validation-only调用链与原默认路径；Ruff作业171808检查通过。新测试为`verl/tests/trainer/ppo/test_standalone_validation_on_cpu.py`，已加入共享快照清单。Ruff安装在独立工具目录，没有改训练环境。

短验证只检查加载、显存、交互和协议能否运行，不用其任务分数选择配置。正式报告只使用正式80条轨迹，不能混入短验证样本。每次提交的具体参数与代码身份保存在对应目录的`submission.json`，实际有效配置保存在`eval/run.log`。

## 对照与结论边界

相同test10×8设置下，E00原始Qwen3-8B为13/80（16.25%），E01 step200为52/80（65.00%）。Qwen3-32B的结果须待作业结束且`eval/summary.json`验收后填写。本次使用test是用户明确要求的模型比较，不据此选择训练超参数或重新选择E06 checkpoint。

本次由explorer审查共享入口、协议和显存路径，worker实现独立推理接入并完成CPU检查，主线程核对资源、模型、成功判定与源码manifest并提交。尚未对训练算法做修改。
