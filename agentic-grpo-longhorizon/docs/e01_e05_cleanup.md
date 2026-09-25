# E01 / E05 目录清理记录

2026-09-25 按用户要求清除早期测试、验证和未完成的旧正式运行，共 24 个目录，移除约 64.35 GiB SSD 分配空间。清理前无活跃 Slurm 作业；explorer 只读审计依赖，主线程执行删除与验收。

## 保留

- E01：`formal-seed42-200-v3/`、`source-formal-v4/`、`test-step200-seed42-v1/`。
- E05：`formal-seed42-200-v1/`、`source-v1/`。保留该正式实验的原验收失败状态。
- 两组第 50/100/150/200 步 HDD 正式 checkpoint，共 8 份；模型与环境未删除。
- 既有 W&B 曲线、上传回执和正式训练日志。
- `github-worktree/` 为仓库同步工作树，`tools/` 为共享 Ruff 工具，不属于测试实验。
- 顶层 submission、source manifest、CPU 验证等小型溯源 JSON。

E01 最终 test 的 eval 符号链接依赖 source-formal-v4 内的 run-167648，因此保留整个最终源码快照；没有改名或搬移正式运行。

## 清理及历史引用

精确目录名单、删除前小型运行摘要、完成时间见 [清理清单](e01_e05_cleanup_manifest.json)。旧计划和历史 submission 中的测试路径可能不再存在，不能据此断言当前数据丢失；正式结果路径不变。Git 当前版本同步移除了对应的 16 个旧测试产物文件，历史提交仍可追溯。

清理后验证：两份正式 metrics/summary 存在，全部 8 个正式 checkpoint 链接及 LoRA adapter 可访问，E01 held-out test summary 可访问。
