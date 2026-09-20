# Qwen3 实验结果

验收通过：True；协议：airline-policy-initial-user-v2。

| 划分 | 任务数 | 轨迹数 | Pass@1（宏平均成功率） | Pass^4（四次全成功） | Pass@4（至少一次成功） |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 40 | 96 | 18.75% | 2.02% | 37.86% |

训练轨迹的指标仅用于诊断；模型效果以固定 checkpoint 的独立评测为准。未完成运行不构成可比较结果。

失败检查：[]。
读取错误：[]。

学习信号：{'groups': 12, 'mixed_outcome_groups': 5, 'signal_observed': True}。
