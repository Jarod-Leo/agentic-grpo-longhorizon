# E01 MiMo中断故障分析与修复

## 已确认的直接原因

正式作业运行4小时17分25秒，完成74次参数更新，在step75的用户模拟器调用中终止。失败请求trajectory_id=97156005f38f4a4380a497742c2e6fbf，task_id=28；api.jsonl记录attempt=1、usage=null、retry=false、耗时0.276秒。异常栈明确落在MimoClient.complete的本地响应校验，最外层异常为UserSimulatorAPIError。

旧代码先读取usage，再判断回复是否有效。usage=null时立即抛出UserSimulatorAPIError；本地校验异常不属于429/5xx/连接超时，因此不重试并设置共享fatal，整个rollout批次失败。通用异常包装丢失了“missing token usage”原因，并把已收到响应的status也清为null。由此可以确认触发点是usage缺失，不能由现存日志断定该次响应文本是否正常，也不能推断服务端为何没有返回usage：旧日志没有保存finish_reason、content长度或响应结构。

[MiMo官方非流式接口定义](https://mimo.mi.com/docs/en-US/api/chat/openai-api)将usage声明为object或null。因此usage缺失本身不能视作对话失败，也不应仅为补计费信息而重新采样用户回复。

## 排除与其他发现

- GPU未OOM；最后一次成功更新grad_norm=0.0834，框架历史显存reserved峰值81.07GiB，Slurm退出原因是Python异常。
- MiMo峰值滚动请求数100RPM，与配置上限一致。其余4次网络类失败在重试后继续运行，并非最终中断原因；旧日志缺少异常类型，不能还原其具体网络原因。
- 还有一个task28轨迹在step59生成了5段非空think标签。非思考模板断言未失败，属于生成内容的协议偏离，不是此次终止原因。保留原审计和严格验收，未屏蔽这些输出，也未改变采样或奖励来隐藏问题。
- checkpoint计划只有100/200步，失败前没有保存点；本轮74步权重无法恢复。HDD目录为空并非归档失败。

## 作业163430后的首轮修复（source-formal-v3）

1. 先校验choices、finish_reason与非空文本；有效stop回复允许usage缺失，直接使用原回复，不额外采样。
2. 缺失或无效usage保持未知；限流器保留原保守token reservation，不能按0计费。用户模拟器总费用返回None，汇总增加usage_complete与requests_missing_usage，已知token总量明确只是已报告部分。
3. 空文本、缺失choices、length/repetition_truncation等异常最多尝试4次，同一消息历史不变；仅成功回复追加到用户历史。content_filter、tool_calls、认证/参数错误不盲目重试。所有最终失败仍中断，不能转为任务失败奖励。
4. 日志新增response_received、finish_reason、content_chars、error_type、error_reason、usage_missing；不记录凭据、原始SDK异常文本或请求头。
5. 用户批准恢复checkpoint改为50/100/150/200步保存并归档HDD，评测仍在100/200步（40个train任务×8次）。不改模型、训练任务、采样参数、优化器、动态微批32768或奖励。

## 验证与重新运行

新增故障注入覆盖：合法usage=null一次成功且费用未知；空回复/截断/缺失choices重试恢复；连续4次异常后中止；content_filter不重试；429/503重试、401立即停止；日志不泄露原始异常内容；缺失usage维持保守限流。测试使用真实OpenAI SDK响应类型，不调用在线API。

随后以冻结的新源码运行1步、32轨迹的真实MiMo+GPU smoke并保存归档checkpoint；只有smoke成功，新的200步正式作业才能启动。旧失败目录与快照保留，正式训练从原始Qwen3与seed42重新开始。原74步失败运行不用于模型效果比较。

CPU预检164674通过：42 tests passed，完整配置/tokenizer与7040轨迹检查通过。真实GPU smoke作业 **164680**；正式重跑作业 **164681** 依赖smoke成功，否则自动取消。源码快照source-formal-v3，输出目录分别mimo-fix-smoke-v1及formal-seed42-200-v2。


## 2026-09-22：作业164681在step4中断

上一轮真实GPU验证164680已经成功：1步、32条轨迹、验收通过，完整checkpoint归档到HDD。随后正式作业164681运行27分57秒，完成3次更新，在step4的rollout阶段失败；未到第50步，因此没有正式checkpoint。

新增诊断确定了本次具体原因：`formal-seed42-200-v2/train/api.jsonl`第705行，trajectory `a1e9d0f8a7ba45d0b2209fe0b9a33e81`、task 8，在policy第11轮输出后的用户模拟器调用中，MiMo返回HTTP 200、`finish_reason=content_filter`、`content_chars=60`、`usage=null`。客户端将该结束原因设为不可重试，第一次即设置共享fatal并终止整个批次。GPU显存不是此次报错源，usage缺失也不再是本地拒绝原因。

[MiMo官方接口文档](https://mimo.mi.com/docs/en-US/api/chat/openai-api)说明content_filter表示内容因过滤而省略。日志没有保存该响应正文，不能推断60个字符是什么，也不能断言具体哪句话触发了过滤；它没有被追加到用户历史。

### 本次最小修复

- content_filter进入已有的有限重试路径：同一请求最多4次（首试加3次重试），消息、模型和采样参数保持不变，继续服从100RPM/1000万TPM和退避。
- 仅接受非空、finish_reason=stop的有效回复。过滤事件保留在API日志中，独立标记error_reason=content_filter；不把过滤文本加入对话、不伪造回复或奖励、不丢弃任务或重采整组轨迹。
- 四次均失败仍明确中止；认证/参数等不可恢复错误继续立即中止。此改动防止一次过滤直接杀死作业，不证明服务端过滤一定是暂时的，也不保证持续过滤能够恢复。
- 保留严格汇总验收与4任务×8轨迹的GRPO分组，训练、评测复用同一个客户端。过滤重试会增加请求数与可能的费用，后续实验和基线比较应记录该客户端版本与过滤频率。

验证覆盖过滤后成功、连续过滤耗尽、请求参数与历史不变、无效文本不进入用户历史、四次失败后共享中止，以及原有usage缺失、429/503、401等路径。实际SDK类型用于离线响应构造，不为测试主动触发在线过滤。

模型、环境、GPU更新路径与归档实现未改动，复用已通过的164680单卡smoke；本次重新运行共享CPU预检后提交正式训练，不重复申请一份相同GPU smoke。新快照为source-formal-v4，正式输出为formal-seed42-200-v3；200步、seed42、动态微批32768、checkpoint每50步归档HDD、train评测每100步。失败作业没有可匹配的正式恢复点，新作业从基础模型重新开始，历史快照与日志保留。


本次验收结果：CPU作业 **165404** 成功，43 tests passed，配置/tokenizer与7040条轨迹检查通过；两处改动的Ruff check和format检查通过。正式重跑 **165407** 已提交，单PRO6000、36小时上限，使用source-formal-v4及formal-seed42-200-v3；提交后首次状态为PENDING (Priority)，不代表训练已经完成。HDD归档目录为 `/projects/_hdd/cabinagentrlarchive/CabinAgent-RL/checkpoints/e01_vanilla_grpo/formal-seed42-200-v3`，预计在50/100/150/200步生成完整checkpoint。

本次实际分工：explorer定位失败轨迹与官方接口语义；worker实现客户端最小修复和两个回归用例；主线程审查、执行Slurm预检、核对资源与归档并重提正式作业。快照中保留了patch工具生成的两个不参与导入执行的`.py.orig`备份；部署副本备份已清理，运行快照保持不可变。
