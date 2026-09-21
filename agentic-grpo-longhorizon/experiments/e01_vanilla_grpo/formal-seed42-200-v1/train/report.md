# Qwen3 实验结果

验收通过：False；协议：airline-policy-initial-user-v2。

| 划分 | 任务数 | 轨迹数 | Pass@1（宏平均成功率） | Pass^4（四次全成功） | Pass@4（至少一次成功） |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 40 | 2371 | 20.53% | 5.68% | 42.01% |

训练轨迹的指标仅用于诊断；模型效果以固定 checkpoint 的独立评测为准。未完成运行不构成可比较结果。

失败检查：['process_success', 'expected_unique_trajectories', 'thinking_disabled', 'audit_covers_trajectories', 'api_no_terminal_failure', 'training_steps', 'groups_per_step', 'actor_updates_per_step', 'evaluation_steps', 'checkpoint_100_complete', 'checkpoint_200_complete']。
读取错误：[]。

学习信号：{'groups': 298, 'mixed_outcome_groups': 148, 'signal_observed': True}。
