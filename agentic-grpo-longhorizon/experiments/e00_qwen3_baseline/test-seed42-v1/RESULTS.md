# E00 原始 Qwen3-8B：独立 test 评测

作业 `170960` 已完成（COMPLETED，exit 0:0），耗时 13 分 43 秒；summary 全部检查通过。

同 E01 test 的冻结评测源码、10 个任务、每题 8 次、seed 42、MiMo 用户模拟器，原始模型不加载 adapter。

| 指标 | E00 base | E01 step200 |
| --- | ---: | ---: |
| 成功轨迹 | 13/80 | 52/80 |
| pass¹ | 16.25% | 65.00% |
| pass⁴（四次全成功） | 0.142857% | 60.00% |
| pass@4（四次至少一次成功） | 47.00% | 75.714286% |

同协议 test 的 pass¹ 提升 48.75 个百分点。任务只有 10 个，结果描述该留出集，不能视为官方全量 leaderboard 成绩。pass⁴/pass@4 为每题 8 次结果的组合估计，再按任务宏平均。

工具调用 schema 严格合法率为 757/773=97.93%；MiMo 请求 411 次全部成功，无重试和终端 API 失败。65 条轨迹达到 assistant turn 上限，15 条由交互结束；终止原因与环境成功分数是两个不同字段。

原始结果见 [summary.json](eval/summary.json) 和 [逐任务结果](eval/task_results.jsonl)。
