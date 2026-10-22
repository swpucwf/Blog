# Streaming：实时观察主 Agent、子 Agent 与工具

## 1. 一次「看起来卡住」的请求

研究助手跑了两分钟。后台日志显示它调了搜索、也确实启动了 `researcher`；浏览器里的用户只看到转圈。等到最终答案出现，用户不知道它是在工作，还是已经卡死。

原始调用没有错，甚至很适合脚本：

```python
result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
print(result["messages"][-1].content)
```

但 `invoke()` 只有整个运行结束才返回，页面无法回答最基本的问题：

- coordinator 有没有真的把任务交给 `researcher`？
- 子 Agent 正在搜索，还是已经失败？
- 工具参数还在生成，还是工具已经返回错误？
- 最终答案出现前，到底发生了哪些步骤？

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_44-comparison-invoke-vs-streaming.png

<p>图 1：invoke 与 Streaming 的体验对比</p>

## 2. 先让案例稳定复现

完全依赖模型自由发挥时，它有时直接回答、不调用 `task`，页面就没有稳定的子 Agent 可展示。为了排查显示链路，先把研究请求固定委派给 `researcher`：

```python
agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a coordinator. Delegate every research request to the "
        "researcher subagent. Do not research the topic yourself. "
        "After the subagent returns, summarize its result in two sentences."
    ),
    subagents=[{
        "name": "researcher",
        "description": "Researches a topic and returns a concise summary.",
        "system_prompt": "Research the topic, use available tools when useful, and return a concise evidence-aware summary.",
    }],
)

request = {"messages": [{"role": "user", "content": "Research recent Agent streaming patterns"}]}
```

这时有了可观察的树：顶层 coordinator，下面一次 `researcher` 委派，研究过程中还可能有工具调用。模板里的搜索工具与模型必须支持 Tool Calling，否则只能看到 Agent 状态，看不到工具事件。

## 3. 修复一：显示「研究助手已启动」

### 3.1 不要从 graph node 猜产品状态

底层 LangGraph 会产生 `model_request`、`tools` 等节点名，它们适合调试，不适合界面——用户关心「研究助手」，不是内部节点。

Deep Agents 为委派提供更直接的视图 `stream.subagents`：每个 handle 对应一次 `task` 委派，带名称、路径与生命周期状态。

```python
stream = agent.stream_events(request, version="v3")

for subagent in stream.subagents:
    print(f"[{subagent.name}] {subagent.status}")
    print("path:", subagent.path)
    try:
        print("output:", subagent.output)
    except Exception as exc:
        print(f"failed: {exc}")
```

handle 刚出现时 `status` 是 `started`，页面可渲染成 `researcher · running`。成功时 `output` 是最终状态；失败或被中断时读取可能抛异常，应成为卡片上的错误，而不是被吞掉。

### 3.2 拆解 handle

| 字段/projection | 含义 | 使用 |
|---|---|---|
| `name` | coordinator 调用 `task` 时选的 `subagent_type` | 显示角色名，不作唯一键 |
| `path` | 这次委派在 Agent 树中的 namespace 路径 | 当前运行内的卡片路由键 |
| `status` | `started`/`completed`/`failed`/`interrupted` | 状态徽标；终态仍要结合 `output`/异常 |
| `messages` | 子 Agent 的消息 | 写入对应卡片 |
| `tool_calls` | 子 Agent 的工具调用 | 卡片下建工具行 |
| `values` | 状态值快照 | 调试/状态面板，不等同文本 |
| `subagents` | 嵌套委派 | 递归建子卡片 |
| `output` | 最终状态或完成信号 | 完成后保存，失败时捕获异常 |

`name` 不是 graph node 名，也不是运行唯一 ID，而来自 `subagent_type`；同一个 `researcher` 可被调用多次，所以 UI 按 `path` 存状态、用 `name` 显示标签。

`status` 是生命周期，不是质量结论：`completed` 只代表这次委派正常结束，内容对不对仍要靠测试、Rubric 或人工审核。

### 3.3 path、namespace、ns 的关系

三个名字是同一概念——从根 Agent 到当前执行位置的 namespace 路径，只是出现在不同 API 层：

| API | 字段 | 形状 | 表示 |
|---|---|---|---|
| v3 Typed Projection | `subagent.path` | `tuple[str, ...]` | 子 Agent 委派的根路径 |
| v3 raw protocol | `event["params"]["namespace"]` | `list[str]` | 产生事件的完整路径 |
| v2 Streaming | `chunk["ns"]` | `tuple[str, ...]` | 产生 chunk 的完整路径 |

每段采用 `<node_name>:<runtime_or_task_id>`。通过 `task` 启动一级子 Agent 时，`subagent.path` 通常只有一个 `tools:<id>` 段——`tools` 是承载委派的图节点，`<id>` 区分本次执行，不是 `researcher` 这样的显示名。

```text
根 Agent 的 v2 ns              ()
根 Agent 的 v3 raw namespace   []
一级子 Agent 的 path            ("tools:abc123",)
子 Agent 内部模型事件的 v2 ns   ("tools:abc123", "model_request:def456")
```

子 Agent 根级事件归一化后通常等于 `subagent.path`；更深层事件以它为前缀：

```python
def belongs_to_subagent(namespace, path) -> bool:
    return tuple(namespace[: len(path)]) == path
```

同名 `researcher` 可多次调用，每次 `<id>` 不同；UI 用完整 `path` 作键，不要用 `name` 代替路径，也不要要求每个事件的 namespace 与根 `path` 完全相等。`path` 的唯一性边界是**当前运行**；写入数据库或 Trace 时，应用要先建立自己的请求记录，再把 `path` 作为其下分支键。

### 3.4 Projection 按需打开

在 deepagents 0.7.15 / langgraph 1.2.11 中，`messages`、`tool_calls`、`subagents` 等图级通道**按订阅消费**：只有订阅后的事件才进入该消费者的缓冲。仅访问属性不等于订阅；同步代码用 `iter(...)`、`for` 或 `interleave(...)` 订阅。

这对生产页面很实用：只展示状态和耗时时，不必消费每条消息与工具增量。但要观察的 projection 应在事件到达前订阅——先读 `stream.output` 推动图到结束、再订阅 `stream.subagents`，可能已经收不到委派 handle。

父级 `stream` 提供同方向的 projection：`messages`（coordinator 消息）、`tool_calls`、`values`、`subagents`、`output`。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_45-framework-typed-projections.png

<p>图 2：Typed Projection 的层级与作用域</p>

最容易犯的错是范围混淆：`stream.messages` 不合并子 Agent 消息，`subagent.messages` 也不含 coordinator 的最终总结。

`values` 不是另一种消息流，也没有统一业务字段——它是当前 Agent state 的快照，字段取决于应用与中间件。调试与状态面板适合；主对话文本仍从 `messages` 读，工具结果仍从 `tool_calls` 读。v0.7 默认不再启用 Todo，前端若靠 `todos` 渲染进度，必须在 Agent 端显式传 `TodoListMiddleware`，并让「没有该字段」成为合法状态。

不要把每个 `value` 当可追加的增量：它更像「这一时刻的快照」，页面按 Agent 路径覆盖最新即可；需要持久化/重放时用 raw event 的 `seq` 排序，`timestamp` 只适合展示。

## 4. 修复二：让用户知道在查什么

```python
with agent.stream_events(request, version="v3") as stream:
    for kind, item in stream.interleave("messages", "subagents"):
        if kind == "messages":
            print("[coordinator]", str(item.text))
        else:
            for message in item.messages:
                print(f"[{item.name}]", str(message.text))
```

### 4.1 message 读什么

Typed Projection 已把底层 content-block 事件整理成 message handle，本例只需要 `text`。同步 `message.text` 也是文本 projection：`str(message.text)` 会等待当前消息结束取完整文本，异步用 `await message.text`；逐段显示可迭代文本投影增量，要保留所有层级的精确协议顺序则用 raw events。

消息对象解决「显示什么」，「属于谁」由消费的 projection 决定。官方 handle 没有 `source`/`kind` 字段；模板的 `event_adapter.py` 才转成：

```python
{"kind": "message", "source": "coordinator", "path": [], "text": message.text, "final": False}
```

`kind` 是模板写入的判别字段，`source` 是模板根据当前 projection 显式传入的标签，都不是框架自动附带的协议字段。子 Agent 消息用 `source="subagent"` 并把 `subagent.path` 写入 `path`。**不要只把文本字符串推给前端**，否则两个来源交错时无法还原归属。

这个同步示例在发现子 Agent 时立即消费其消息，避免先耗尽主流、再晚订阅子 Agent——后者在当前版本可能直接漏掉 handle，而不只是顺序变化：

```text
[coordinator] 正在委派研究任务
[coordinator] 这是最终总结……
# 子 Agent 确实执行了，但消费者没有及时订阅到它。
```

但 `str(item.text)` 会等完整消息，内层循环也会暂停外层消费，因此不能保证实时页面及时展示所有交错增量。

## 5. 修复三：工具调用也要可见

```python
with agent.stream_events(request, version="v3") as stream:
    calls = []
    for kind, item in stream.interleave("tool_calls", "subagents"):
        if kind == "tool_calls":
            calls.append(("coordinator", item))
        else:
            for call in item.tool_calls:
                calls.append((item.name, call))
                print(f"[{item.name} tool]", call.tool_name, call.input)
                for delta in call.output_deltas:
                    print(delta, end="", flush=True)

    final_state = stream.output
    for source, call in calls:
        print(f"[{source} tool]", call.tool_name, call.input)
        if call.completed and call.error is None:
            print("result:", call.output)
        elif call.error is not None:
            print("error:", call.error)
```

### 5.1 拆解 tool_call

| 字段/projection | 含义 | 规则 |
|---|---|---|
| `tool_name` | 模型选的工具名 | 显示与分类；未知工具名不可信 |
| `input` | 工具输入 | 可能含敏感数据，日志前先脱敏 |
| `output_deltas` | 执行中的增量输出 | 按到达顺序追加，单个 delta 不是完整结果 |
| `completed` | 是否进入完成状态 | `True` 仍要检查 `error` |
| `output` | 成功调用的最终输出 | `completed` 且 `error is None` 后读取 |
| `error` | 调用错误 | 非空进失败分支 |

状态不能只写 `if call.completed: success`：

```python
if not call.completed:
    status = "running"
elif call.error is not None:
    status = "failed"
else:
    status = "completed"
```

`output_deltas` 适合终端/实时日志，`output` 适合完成后摘要；两者都留时避免重复（不要先把 delta 拼全文、再把相同 output 追加一次）。

补上工具状态后，卡片才真正有信息：

```text
researcher · running
  search · running   query="Deep Agents event streaming"
  search · completed 5 results
  正在整理 v3 projection 与 v2 protocol 的差异……
```

嵌套委派可递归：遍历 `subagent.subagents`，每层用 `path` 作 UI 唯一键——同名 `researcher` 可能来自不同分支，只用 `name` 会把两张卡片写到一起。

## 6. 顺序乱了：两种实时消费方式

### 6.1 异步服务：并发消费

coordinator 与 researcher 的事件交错到达，实时 UI 不能把两个 iterator 排队处理：

```python
async def stream_live():
    stream = await agent.astream_events(request, version="v3")

    async def consume_coordinator():
        async for message in stream.messages:
            print("[coordinator]", await message.text)

    async def consume_subagents():
        async for subagent in stream.subagents:
            async for message in subagent.messages:
                print(f"[{subagent.name}]", await message.text)

    await asyncio.gather(consume_coordinator, consume_subagents)
```

`asyncio.gather` 解决阻塞：一个投影等网络时，另一个仍能把事件送到页面。它不返回带全局序号的合并 iterator；要审计「哪个 token 先到」，仍需 raw event 的 `seq`。

### 6.2 同步程序：interleave

```python
for name, item in stream.interleave("messages", "subagents"):
    if name == "messages":
        print("[coordinator]", item.text)
    else:
        for message in item.messages:
            print(f"[{item.name}]", message.text)
```

`interleave` 按严格到达顺序合并**显式选择**的 projections，每次返回 `(name, item)`。没传 `tool_calls`，coordinator 的工具调用就不会出现在循环里；子 Agent 工具仍要从 `item.tool_calls` 读。

它适合快速做同步展示，但读取完整文本或进入内层循环仍可能阻塞外层，也不能保证整棵树的 token/嵌套顺序；实时异步应用应并发消费，精确协议排序用 raw `seq`。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_46-sequence-concurrent-streaming.png

<p>图 3：并发消费与顺序边界</p>

## 7. 需要精确顺序时才读 raw events

大多数产品只需要「主对话」「researcher 卡片」「工具行」三个区域；只有调试丢事件、重放或审计时，才值得保留所有层级的精确到达顺序：

```python
stream = agent.stream_events(request, version="v3")

for event in stream:
    if not isinstance(event, dict) or event.get("method") != "messages":
        continue
    params = event.get("params") or {}
    data = params.get("data")
    if not isinstance(data, (list, tuple)) or not data:
        continue
    payload = data[0]
    if not isinstance(payload, dict) or payload.get("event") != "content-block-delta":
        continue
    block = payload.get("delta") or {}
    if block.get("type") != "text-delta":
        continue

    namespace = params["namespace"]
    source = "subagent" if namespace else "coordinator"
    print(f"#{event['seq']} [{source}] {block['text']}", end="", flush=True)
```

raw event 字段属于协议层；上例的 `source` 只是按 `namespace` 算出的本地标签，不是事件自带字段。建议集中写一个 adapter：校验版本、读 `seq`/`namespace`，再转成应用自己的事件格式，页面只消费转换后的对象。

> 过滤器只读 content-block 文本增量。没有原生分块输出的模型可能发出完整消息，过滤结果为空不等于图没有输出；解析 `params.data` 要结合实际模型与锁定版本。

| 路径 | 形状 | 处理 |
|---|---|---|
| `event["seq"]` | int | 同一次运行内严格递增，排序与查漏用它，不用时间戳 |
| `event["method"]` | str | 先按方法分派，再读 payload |
| `params["namespace"]` | list[str] | 图层级路径；空表示根，非空用于子图路由 |
| `params["timestamp"]` | int | 展示用，时钟可能漂移，不用于严格排序 |
| `params["data"]` | Any | 形状由 method 决定，确认是 messages 后再按 content-block 解析 |

`namespace` 不是显示名：v3 的 `subagent.name` 来自 `subagent_type`，raw namespace 是完整执行路径。前端 adapter 可同时保存 `name`、`path`、raw `namespace`，分别用于显示、卡片路由与协议审计。

## 8. 旧代码为什么还在处理 type/ns/data

接手已有 LangGraph 服务时常看到：

```python
for chunk in agent.stream(request, stream_mode=["updates", "messages", "custom"],
                          subgraphs=True, version="v2"):
    print(chunk["type"], chunk["ns"], chunk["data"])
```

v2 的每个 `StreamPart` 有 `type`、`ns`、`data`：

| 字段 | 形状 | 解读 |
|---|---|---|
| `type` | str | `updates`/`messages`/`custom` |
| `ns` | tuple[str, ...] | `()` 主 Agent；非空是完整 namespace |
| `data` | 随 type 变 | updates 常是节点状态，messages 是消息与 metadata，custom 是自定义 payload |

```text
()                              -> main agent
("tools:abc123",)              -> task 启动的子 Agent
("tools:abc123", "model_request:def456") -> 子 Agent 内部模型节点
```

按 `tools:` 段路由子 Agent 更新：

```python
for chunk in agent.stream(request, stream_mode="updates", subgraphs=True, version="v2"):
    if chunk["type"] != "updates":
        continue
    ns = chunk["ns"]
    task_segment = next((s for s in ns if s.startswith("tools:")), None)
    if task_segment is None:
        print("Main agent:", chunk["data"])
    else:
        print(f"Subagent {task_segment.split(':', 1)[1]}:", chunk["data"])
```

嵌套时可能多个 `tools:` 段，应保留完整 `ns` 或按已知 `subagent.path` 前缀匹配，不能只取第一个 ID 当全局唯一键。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_47-comparison-v3-v2-streaming.png

<p>图 4：v3 与 v2 的观察层级</p>

`stream_mode` 决定 `data` 形状：`updates` 看节点状态，`messages` 看 token 与工具消息，`custom` 看应用进度；`subgraphs=True` 才让子图事件进入同一条流。`messages` 分支不能只打印 tuple，要区分 `AIMessageChunk`（含 `tool_call_chunks`）、`ToolMessage` 与普通文本，来源结合 `chunk["ns"]`。

这套格式适合两类场景：已有应用迁移、不想一次重写事件路由；调试时必须知道更新来自哪个图节点。新页面仍建议 v3 的 `subagents`——直接表达「委派给哪个产品角色」，不让 UI 猜 namespace。

## 9. 自定义进度

工具要报告「已找到 3 个来源」「正在合并摘要」，应显式发出：

```python
from langgraph.config import get_stream_writer

@tool
def analyze_data(topic: str) -> str:
    """Analyze a topic and report structured progress."""
    writer = get_stream_writer()
    writer({"status": "starting", "topic": topic, "progress": 0})
    writer({"status": "complete", "topic": topic, "progress": 100})
    return f"Analysis complete: {topic}"
```

v2 从 `custom` 分支读；v3 可按业务 schema 扩展 adapter。当前模板没有定义 `kind="progress"`，不要把示例 payload 当成框架固定事件；也不要让前端一半读 v3 对象、一半判断 v2 字段。

custom payload 由应用定义，应加版本与验证：`status` 用有限枚举、`topic` 只在必要事件发送、`progress` 约定 0-100 且只表示业务估算；生产事件加 `schema_version`，未知版本记录并降级。

## 10. 从「能看到」到「能交付」

页面终于能把一次研究说清楚：

```text
coordinator message  -> 主对话
subagent started     -> researcher 卡片 running
subagent message     -> 写入 researcher 卡片
tool call            -> 卡片里的工具行（参数与结果）
subagent completed   -> 卡片收起，保留摘要与状态
final output         -> 主对话最终答案
error/interrupted    -> 对应层级的错误或中断提示
```

`deepagents/streaming` 模板已在 `event_adapter.py` 把 typed projections 转成 SSE 事件：`message`（source/path/text/final）、`subagent`（phase/name/path/status）、`tool_call`（phase/source/path/tool_name/input/delta/output/error）、`values`、`output`、`raw`（sequence/method/namespace/data）。

`kind` 是模板写入的字面量、`source` 是模板传入的标签、`phase` 是模板处理阶段——都不是框架字段；`seq`/`method`/`params` 才属于 raw protocol。这里没有 `run_id`：模板请求只有可选的 `thread_id`，用于保持会话连续性，不是单条流事件 ID。

### 四个生产问题

1. **Streaming 不等于并行**：是否并行取决于编排方式，不能从事件交错反推执行模型；
2. **客户端断开**：长任务要明确选超时、取消，还是转后台继续；
3. **慢客户端**：服务端限制队列大小，或丢弃可重建的增量，不能无限积压；
4. **刷新后能否重看**：需要重放就把标准化事件写入 Trace、数据库或对象存储；内存 iterator 不是持久化日志。

工具错误、子 Agent 失败与中断状态都要原样保留——只有把失败也做成事件，用户才知道「研究没完成」和「页面没刷新」是两回事。

## 11. 本篇验收

- [ ] 能解释 `invoke()` 为什么让页面像卡住；
- [ ] 会用 `stream.subagents` 显示委派状态与 `path`；
- [ ] 知道 `name` 来自 `subagent_type`，`path` 才是运行内唯一键；
- [ ] 理解 projection 按订阅消费、要提前订阅；
- [ ] 能读 `message.text` 与 `tool_call` 的 completed/error/output；
- [ ] 会用 `asyncio.gather` 与 `interleave` 解决交错；
- [ ] 知道精确顺序要看 raw `seq`/`namespace`；
- [ ] 能读懂 v2 的 `type/ns/data` 并知道何时该用 v3；
- [ ] 会用 `get_stream_writer` 发自定义进度；
- [ ] 记住 Streaming 不负责持久化、取消、超时与背压。

> **补记**：Streaming 的改造顺序比 API 本身更重要——**先显示「谁在工作」，再显示「在做什么」，最后才处理精确顺序**。三个高频坑：晚订阅导致漏掉子 Agent handle、把 `stream.messages` 当全量消息、把模板的 `kind/source/phase` 当成框架协议字段。生产上还要补上断开、慢客户端与重放策略。

## 写在最后

下一篇[Interpreters 代码编排](16-Interpreters代码编排)把循环与批处理移进代码：QuickJS 内执行 JavaScript，用 PTC 批量调用工具，只把整理后的结果带回对话。
