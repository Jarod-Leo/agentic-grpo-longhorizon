# E01 作业163430故障分析与修复

## 已确认的直接原因

正式作业运行4小时17分25秒，完成74次参数更新，在step75的用户模拟器调用中终止。失败请求trajectory_id=97156005f38f4a4380a497742c2e6fbf，task_id=28；api.jsonl记录attempt=1、usage=null、retry=false、耗时0.276秒。异常栈明确落在MimoClient.complete的本地响应校验，最外层异常为UserSimulatorAPIError。

旧代码先读取usage，再判断回复是否有效。usage=null时立即抛出UserSimulatorAPIError；本地校验异常不属于429/5xx/连接超时，因此不重试并设置共享fatal，整个rollout批次失败。通用异常包装丢失了“missing token usage”原因，并把已收到响应的status也清为null。由此可以确认触发点是usage缺失，不能由现存日志断定该次响应文本是否正常，也不能推断服务端为何没有返回usage：旧日志没有保存finish_reason、content长度或响应结构。

[MiMo官方非流式接口定义](https://mimo.mi.com/docs/en-US/api/chat/openai-api)将usage声明为object或null。因此usage缺失本身不能视作对话失败，也不应仅为补计费信息而重新采样用户回复。

## 排除与其他发现

- GPU未OOM；最后一次成功更新grad_norm=0.0834，框架历史显存reserved峰值81.07GiB，Slurm退出原因是Python异常。
- MiMo峰值滚动请求数100RPM，与配置上限一致。其余4次网络类失败在重试后继续运行，并非最终中断原因；旧日志缺少异常类型，不能还原其具体网络原因。
- 还有一个task28轨迹在step59生成了5段非空think标签。非思考模板断言未失败，属于生成内容的协议偏离，不是此次终止原因。保留原审计和严格验收，未屏蔽这些输出，也未改变采样或奖励来隐藏问题。
- checkpoint计划只有100/200步，失败前没有保存点；本轮74步权重无法恢复。HDD目录为空并非归档失败。

## 最小修复

1. 先校验choices、finish_reason与非空文本；有效stop回复允许usage缺失，直接使用原回复，不额外采样。
2. 缺失或无效usage保持未知；限流器保留原保守token reservation，不能按0计费。用户模拟器总费用返回None，汇总增加usage_complete与requests_missing_usage，已知token总量明确只是已报告部分。
3. 空文本、缺失choices、length/repetition_truncation等异常最多尝试4次，同一消息历史不变；仅成功回复追加到用户历史。content_filter、tool_calls、认证/参数错误不盲目重试。所有最终失败仍中断，不能转为任务失败奖励。
4. 日志新增response_received、finish_reason、content_chars、error_type、error_reason、usage_missing；不记录凭据、原始SDK异常文本或请求头。
5. 用户批准恢复checkpoint改为50/100/150/200步保存并归档HDD，评测仍在100/200步（40个train任务×8次）。不改模型、训练任务、采样参数、优化器、动态微批32768或奖励。

## 验证与重新运行

新增故障注入覆盖：合法usage=null一次成功且费用未知；空回复/截断/缺失choices重试恢复；连续4次异常后中止；content_filter不重试；429/503重试、401立即停止；日志不泄露原始异常内容；缺失usage维持保守限流。测试使用真实OpenAI SDK响应类型，不调用在线API。

随后以冻结的新源码运行1步、32轨迹的真实MiMo+GPU smoke并保存归档checkpoint；只有smoke成功，新的200步正式作业才能启动。旧失败目录与快照保留，正式训练从原始Qwen3与seed42重新开始。原74步失败运行不用于模型效果比较。

CPU预检164674通过：42 tests passed，完整配置/tokenizer与7040轨迹检查通过。真实GPU smoke作业 **164680**；正式重跑作业 **164681** 依赖smoke成功，否则自动取消。源码快照source-formal-v3，输出目录分别mimo-fix-smoke-v1及formal-seed42-200-v2。
