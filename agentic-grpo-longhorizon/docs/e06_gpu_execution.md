# E06 纯 OPD：GPU 执行记录

2026-09-25 用户授权执行 E06，原“仅 CPU”阶段限制在本次执行中解除。

## 冻结方案

- 学生：原始 Qwen3-8B＋LoRA，seed42；4 个任务×8 条轨迹/步；动态微批32768 tokens。
- 教师：冻结 Qwen3-32B BF16，revision `9216db5781bf21249d130ec9da846c4624c16137`。
- 纯 OPD：RL系数0，蒸馏系数1；采样token PG目标，非全词表KL。
- 同节点两张 Pro6000，同一 srun step，rank0学生、rank1教师，各绑定一张GPU。
- 正式训练200步；每50步保存完整checkpoint并归档HDD；每100步在train40题×8次评测。
- 固定train40/test10；本次不使用test选模型。
- 正式W&B实时上传至 `carrortsky-nanyang-technological-university-singapore/Agentic-longhorizon`。

## 共用入口最小调整

`run_qwen3_teacher_pair.sh`新增可选STUDENT_ENTRY，默认正式入口不变；smoke改用现有run_qwen3.sh，以环境变量指定2步。`qwen3_runtime.sh`尊重外部WANDB_MODE，默认仍disabled，online时核验认证并启用veRL已有wandb logger。凭据仅在student rank从用户指定私有文件加载；不将密钥写入Git或提交记录。

快照：`experiments/e06_opd/source-gpu-v1/`，不修改已提交作业引用的代码。

## 验证与依赖

| 阶段 | Job | 设置 | 状态 |
| --- | --- | --- | --- |
| CPU回归 | 171248 | 96＋59项，共155项；真实tokenizer指纹和配置检查 | COMPLETED，1分4秒 |
| GPU smoke | 171249 | 2步×32条；最终train40×1评测；step2 checkpoint保留SSD | 已启动，待GPU完整验收 |
| 正式训练 | 171250 | 200步，50保存，100评测 | 已提交，afterok:171249 |

smoke使用正式任务batch/长度/算法；只缩短步数及诊断评测重复次数。整个smoke包括教师加载、评分、更新、评测和checkpoint；共享汇总器验收失败会使作业非零退出。正式作业仅在smoke成功退出后运行，失败时依赖取消，不带病启动。

## 存储和资源

HDD归档目录配额在账户已分配的400GB内从350GB提升到400GB；检查时已用309.8GB、余量约90.2GB。预计4份正式checkpoint约64GiB，可容纳；smoke checkpoint留SSD。此前清理后SSD约59.3/150GB，足够active数据和checkpoint暂存。

正式归档：`/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e06_opd/formal-seed42-200-v1/`。

教师目前逐条评分、整条因果forward后分块计算log-prob；分块不会消除forward的完整logits。CPU验证不证明真实GPU显存和吞吐，须以smoke实测为准。正式时限36小时是调度上限，不是已测耗时。

详情见各运行目录的 `submission.json`、`allocation.txt`、rank日志及 `train/summary.json`。
