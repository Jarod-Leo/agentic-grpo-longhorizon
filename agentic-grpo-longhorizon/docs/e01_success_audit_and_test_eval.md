# E01 成功判定审计与固定 test 评测（2026-09-23）

核对对象为真实训练作业165407使用的`experiments/e01_vanilla_grpo/source-formal-v4`，不是仅检查当前开发代码。冻结`tau-bench/tau_bench/envs/base.py`的SHA256为`20f77ff9255ecaa7db0fbd0972a81881393f282766992c94d572e442cddb8a39`，与当前本地基准一致；函数逻辑与Sierra官方τ-bench的`Env.calculate_reward`一致。

成功条件没有被放宽：终止时比较实际数据库状态与按任务参考动作重放所得状态；不相同则0。任务指定outputs时，还检查assistant回答包含所需输出，缺少任意项则0，否则1。MiMo负责用户模拟，不是成功裁判。E01 binary wrapper将环境终局0/1转换为相同二元结果；未用过程奖励或主观LLM评分替代该判定。异常、污染和未成功即耗尽交互预算的轨迹不会被当成成功。

指标`pass¹`为任务宏平均单次成功率；`pass⁴`估计四次全部成功，使用每任务8次样本的C(c,4)/C(8,4)，再跨任务平均；补充`pass@4`表示至少成功一次，两者不能混用。E01第200步的88.75%来自40个train任务各8次、共284/320成功，不是test结果。

协议不与官方榜单完全一致：本项目从原τ-bench airline的50个任务自定义40 train/10 test；MiMo v2.5替代官方默认GPT-4o用户模拟器；veRL限制15 assistant turns和15 user turns，interaction另有30事件限制，不能等同官方30模型动作上限。输入/工具执行走veRL适配。所有比较必须注明这些条件，不能把本项目train或10-task test成绩与其他模型的官方全50任务成绩直接横比。此任务集也不是当前τ³-bench版本。

用户已授权固定最终模型的test评测。提交作业167648，使用原冻结source-formal-v4和共用`scripts/eval/eval_qwen3.sh`，不修改成功判定或任何训练代码。模型为官方Qwen3-8B加E01正式训练第200步HDD LoRA adapter；不按test选checkpoint，不再更新权重。

- test任务：3、10、16、20、23、26、30、36、43、46；每题8次，共80条。
- 采样：temperature=0.7、top_p=0.9、seed=42，与训练内诊断评测保持一致。
- 资源：单Pro6000，2小时上限；复用已通过的单卡adapter评测能力。
- 依赖：afterany:166640，等待E05退出后再使用同一100 RPM MiMo额度。
- 输出：`experiments/e01_vanilla_grpo/test-step200-seed42-v1/eval/`；成功后共用汇总器自动生成summary.json、task_results.jsonl及报告。
- 提交记录：`experiments/e01_vanilla_grpo/test-step200-seed42-v1/submission.json`。提交时状态PENDING/Dependency，尚无测试成绩。

本次由explorer只读审计官方规则、冻结源码及协议差异，主线程核对Slurm资源并提交评测。没有修改评分函数、任务数据或在跑作业。

来源：[官方成功判定](https://github.com/sierra-research/tau-bench/blob/main/tau_bench/envs/base.py)、[官方指标与用户模拟器说明](https://github.com/sierra-research/tau-bench#leaderboard)。

## 2026-09-23 完成记录

作业167648已COMPLETED/0:0，运行18分48秒，summary.accepted=true，所有检查通过。10个test任务各8次，共52/80成功：pass¹=65.00%，pass⁴=60.00%，pass@4=75.7143%。这是固定第200步模型在本项目留出任务上的评测，未更换checkpoint或更新参数。

旧冻结runtime使用默认输出目录source-formal-v4内的experiments/e01_vanilla_grpo/run-167648/eval；现从提交目录test-step200-seed42-v1/eval建立链接指向实际结果，submission.json保存真实路径。旧文中的输出位置现在可通过该链接访问，没有改动冻结源码。
