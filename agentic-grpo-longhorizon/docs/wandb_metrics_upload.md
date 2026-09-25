# E01 / E05 历史训练指标补传 W&B

## 目标与数据来源

- Entity：`carrortsky-nanyang-technological-university-singapore`
- Project：`Agentic-longhorizon`
- E01：`experiments/e01_vanilla_grpo/formal-seed42-200-v3/train/`
- E05：`experiments/e05_prm_lite_lata/formal-seed42-200-v1/train/`

两组原始运行均禁用在线 W&B，使用 veRL file logger 各保存了 200 步 `metrics.jsonl`。补传不重新训练，不改变原始轨迹、奖励或验收结果。

## 认证

标准变量名为 `WANDB_API_KEY`，不是 `WANBD_API_KEY`。其他终端中执行的 `export` 不会自动修改当前代理进程或其他终端的环境。凭据应由用户保存至项目外的私有配置文件，例如 `~/.config/cabinagentrl/wandb.env`，权限设为 `600`；只需告知路径，不要在聊天、日志或 Git 中记录密钥。

## 记录口径

- 原始训练标量保留名称和 global step；补传时间不代表原训练时间。
- `critic/rewards/mean` 是算法实际训练奖励，E05 含过程奖励，不能标为成功率；原始 `best@4/mean` 也不替代汇总器的组合估计 `pass@4`。
- 第 100 / 200 步周期评测属于 **train**，使用 `eval/train/` 指标区分。
- E01 独立 test 的 52/80、pass¹=0.65 记录为第 200 步的 `eval/test/` 指标，不能与 train 成功率混合。
- E05 已训练至 200 步，但 `thinking_disabled` 验收失败；云端必须保留失败标签和检查结果，不能把“指标上传成功”当作“实验验收通过”。
- 不上传模型、对话轨迹、凭据或整个环境；补传机器的系统指标也不作为历史训练性能记录。

## 共用入口

[scripts/eval/upload_metrics_wandb.py](../scripts/eval/upload_metrics_wandb.py) 读取已有 `metrics.jsonl`、`run.json` 和 `summary.json`。使用 `--dry-run` 可以在不认证、不联网的情况下校验待上传数据。正式上传使用 W&B Python SDK；认证缺失时不发起上传。

SDK 的 step 与 summary 语义参见 [W&B Run 官方文档](https://docs.wandb.ai/ref/python/experiments/run/)。本次没有修改原训练启动脚本，因此未来训练是否实时上传仍由对应运行配置决定。

## 本次上传结果（2026-09-25 UTC）

| 实验 | 云端 Run | 历史步数 | 状态 |
| --- | --- | --- | --- |
| E01 | [E01-GRPO-seed42-200](https://wandb.ai/carrortsky-nanyang-technological-university-singapore/Agentic-longhorizon/runs/68bf76aeddf35272) | 1–200 | finished，验收通过 |
| E05 | [E05-PRM-Lite-LATA-seed42-200](https://wandb.ai/carrortsky-nanyang-technological-university-singapore/Agentic-longhorizon/runs/6b166e6f09e31a00) | 1–200 | failed，保留 thinking_disabled 验收失败 |

已通过 W&B API 读回两组各 200 行历史，逐步核对所有原始标量与本地日志一致，并核对 E01 train/test 和 E05 train 评测成绩。`wandb_upload.json` 记录上传回执，`wandb_verification.json` 记录读回校验；均存于各自 train 目录。W&B 自动生成的包版本列表、SDK 元数据与运行时间描述的是补传环境，非原始训练机器；原训练的显存与吞吐指标来自已有 metrics.jsonl。
