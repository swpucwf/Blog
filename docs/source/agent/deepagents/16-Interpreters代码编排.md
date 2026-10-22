# Interpreters：用代码编排工具与数据

## 1. 为什么需要 Interpreter

订单审核 Agent 收到 80 个订单号。它先调用一次查询工具，等结果回来，再调下一次。几十条结果陆续进入上下文，模型还没审完就开始漏项。查询工具没问题，**不合适的是编排方式**：循环、筛选与聚合都交给了模型逐轮决定。

普通 Tool Calling 的循环：

```text
模型决定查询 A-100 -> 工具返回 A-100 -> 结果进入模型上下文
模型决定查询 A-101 -> 工具返回 A-101 -> 结果进入模型上下文
……
模型整理全部结果 -> 返回风险订单
```

模型可以在同一轮发出一批调用，但这批调用在生成完成时就固定了：不能在同一批里读第一个结果再决定第二个，循环、分支与重试通常都要新模型轮次。

数据量小时最直接；变大后问题暴露：调用次数由模型决定（难保证每个输入都被处理）、每个中间结果都进上下文、排序分组去重仍由模型反复做、延迟与成本上升。

| 任务形状 | 优先选择 | 原因 |
|---|---|---|
| 一两个简单外部调用 | 普通 Tool Calling | 路径短，编排无收益 |
| 纯内存排序/分组/解析/校验 | Interpreter | JS 可确定性处理 |
| 大量外部调用、需要循环或并行 | Interpreter + PTC | 代码控制调用与聚合 |
| Shell、装依赖、跑测试、完整文件系统 | Sandbox | 需要独立执行环境与 OS 能力 |
| 大量独立任务需要不同 Agent 角色 | Dynamic Subagents | 每个角色要跑完整子 Agent 推理 |

**Interpreter 不是轻量 Shell，也不是本地沙箱**，而是 Agent 循环里的内存代码运行时。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_48-comparison-tool-calling-vs-interpreter.png

<p>图 1：普通工具调用与 Interpreter + PTC</p>

## 2. 运行第一段 JavaScript

```bash
uv add "deepagents[quickjs]" langchain-openai
```

> Interpreters 目前是 Beta API；示例要求 Python 3.11+、`langchain-quickjs>=0.2.0`，接口与生命周期可能变化。

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model=model,
    system_prompt="Use eval for deterministic filtering and aggregation.",
    middleware=[CodeInterpreterMiddleware(mode="call")],
)
```

`CodeInterpreterMiddleware` 增加一个 `eval` 工具：模型不是直接操作 QuickJS API，而是像调其他工具一样，把自己生成的 JavaScript 交给运行时。模型生成的代码可能不同，核心逻辑类似：

```typescript
const orders = [
  { id: "A-100", amount: 320, refunds: 0 },
  { id: "A-101", amount: 1800, refunds: 0 },
  { id: "A-102", amount: 760, refunds: 3 },
];

const risky = orders
  .filter((order) => order.amount >= 1000 || order.refunds >= 2)
  .sort((left, right) => right.amount - left.amount);

({ ids: risky.map((order) => order.id),
   totalAmount: risky.reduce((sum, order) => sum + order.amount, 0) });
```

`eval` 返回最后一个表达式的值——这里是含 `ids` 与 `totalAmount` 的对象，而不是整个执行过程。`console.log/warn/error` 默认也会被捕获，但更适合调试，不应代替最终表达式。

### 2.1 QuickJS 默认能做什么

| 能力 | 默认可用 | 说明 |
|---|---|---|
| JavaScript 与顶层 `await` | 是 | 循环、分支、Promise、数据转换 |
| `console.*` | 是 | 输出进入 eval 响应 |
| Agent 工具 | 否 | 需显式配置 PTC 白名单 |
| 文件系统 / 网络 / 系统时间 | 否 | 需通过 PTC 暴露具体工具 |
| Shell、包管理器、系统测试 | 否 | 应使用 Sandbox |

这种默认封闭很重要：模型可以写任意 JS，但代码不会因此自动获得宿主机的文件、网络或 Shell 权限。

### 2.2 Interpreter 与 Sandbox 的分工

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_49-framework-interpreter-sandbox-boundaries.png

<p>图 2：Interpreter 与 Sandbox 的边界</p>

| 对比 | Interpreter | Sandbox |
|---|---|---|
| 主要目标 | 循环内编排工具、保存中间值、转换数据 | 隔离环境中操作文件、进程与依赖 |
| 运行时 | 进程内嵌 QuickJS | 独立容器/VM/远程环境 |
| 默认文件/网络 | 无 | 取决于 Backend 与网络策略 |
| 适合 | 批量查询、过滤、排序、重试、聚合 | 编译、测试、装依赖、生成文件 |
| 结果 | eval 结果与捕获的 console | 命令输出、退出码、文件产物 |

把 100 条记录归类用 Interpreter；跑 `pytest` 或装 Python 包用 Sandbox。**不要因为 QuickJS 能运行代码，就把它当安全执行不可信程序的完整隔离环境。**

## 3. 用 PTC 批量调用工具

纯内存示例的数据已经在 Prompt 里。真实应用通常只有订单号，需要从数据库或服务读。此时启用 **PTC（Programmatic Tool Calling）**，把选定工具作为异步函数放进 JS 的 `tools` 命名空间。

先把窄接口定义成 Tool：

```python
@tool
def lookup_order(order_id: str) -> dict:
    """Return one order by ID."""
    order = ORDERS.get(order_id)
    return {"order_id": order_id, **order} if order else {"order_id": order_id, "error": "not_found"}
```

只把这个工具开放给解释器：

```python
agent = create_deep_agent(
    model=model,
    middleware=[CodeInterpreterMiddleware(ptc=[lookup_order], mode="turn", max_ptc_calls=16)],
)
```

直接传 `BaseTool` 对象时，它**只在解释器的 `tools` 命名空间可用**，不会同时成为 Agent 的普通工具——模型必须先调 `eval`，不能绕开批处理逐个查询。

如果工具本来就应同时支持普通调用，改用名称白名单：

```python
agent = create_deep_agent(
    model=model,
    tools=[lookup_order],
    middleware=[CodeInterpreterMiddleware(ptc=["lookup_order"])],
)
```

两种形式不要混淆：名称从 Agent 工具集匹配，对象只暴露给 PTC。

### 3.1 工具名转 camelCase，参数不转

Python 的 `lookup_order` 在 JS 中是 `tools.lookupOrder`，但输入对象仍遵循原 Schema，参数继续叫 `order_id`：

```typescript
const order = await tools.lookupOrder({ order_id: "A-101" });
```

`tools.lookupOrder({ orderId: ... })` 不合法。批处理示例：

```typescript
const ids = ["A-100", "A-101", "A-102", "A-103", "A-104", "A-105", "A-999"];

const rows = await Promise.all(
  ids.map((orderId) => tools.lookupOrder({ order_id: orderId })),
);

const missing = rows.filter((r) => r.error === "not_found").map((r) => r.order_id);
const risky = rows
  .filter((r) => !r.error)
  .filter((r) => r.amount >= 1000 || r.refunds >= 2)
  .sort((a, b) => b.amount - a.amount);

({ riskyIds: risky.map((r) => r.order_id),
   totalAmount: risky.reduce((sum, r) => sum + r.amount, 0),
   missing });
```

七次查询仍真实发生，但模型不需要读七份完整工具消息：QuickJS 等结果、过滤、整理缺失项，只返回一个小对象。

### 3.2 PTC 改变的是调用路径

| 对比 | 普通 Tool Calling | PTC |
|---|---|---|
| 谁控制下一次调用 | 模型下一轮输出 | 当前 eval 的 JavaScript |
| 循环/分支 | 通常需新模型轮次 | 代码内完成 |
| 中间结果 | 每次返回模型上下文 | 留在解释器，整理后返回 |
| 并行批次 | 模型决定一批 | `Promise.all` 等代码决定 |
| 审批路径 | 走普通工具调用机制 | **不逐次走普通审批** |

最后一行是安全边界：**PTC 调用不会逐次执行父 Agent 的 `interrupt_on`**。转账、删库、发信、创建云资源等高副作用工具，不要因为方便就放进白名单。

若业务要求「每次写操作都要人工确认」，保留普通工具路径；若允许一次批准整批流程，可以把 `eval` 作为审批边界，但必须同时限制白名单、最大调用次数与工具自身权限。**恢复解释器状态不会撤销 PTC 已造成的外部副作用**——Snapshot 能恢复 JS 变量，数据库写入、网络请求或付款不会回滚。

## 4. 控制状态保留范围

| `mode` | 状态范围 | 适合 | 风险 |
|---|---|---|---|
| `thread` | 同一线程的多个 Agent turn | 分段分析、跨轮累积 | 旧变量可能影响后续请求 |
| `turn` | 当前 turn 内的多次 eval | 一次请求分步计算 | 下一轮不能复用结果 |
| `call` | 只保留当前一次 eval | 无状态转换、隔离最强 | 多次 eval 要重新准备数据 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_50-timeline-interpreter-state-modes.png

<p>图 3：call、turn、thread 三种状态范围</p>

第一版用 `call`，每次从空环境开始；PTC 版改 `turn`，允许同一轮分多次执行，又不把订单数据带到下一轮。

跨轮才用 `thread`：

```python
interpreter = CodeInterpreterMiddleware(mode="thread", max_snapshot_bytes=4 * 1024 * 1024)
```

`thread` 在 turn 开始时恢复最近快照，结束时把新快照写进 Graph State；一个 turn 内多次 eval 共用同一活动上下文。**Snapshot 只保留可序列化数据**：字符串、数字、数组、普通对象可以；函数、类等恢复后不能继续用——保存计算结果，不要假设动态定义的函数永远存在。

需要保留线程历史或 Time Travel 时加 Checkpointer：

```python
agent = create_deep_agent(
    model=model,
    middleware=[CodeInterpreterMiddleware(mode="thread")],
    checkpointer=MemorySaver(),
)
```

后续每次 `invoke` 复用同一 `thread_id`；`MemorySaver` 只适合当前进程实验，重启后仍要恢复就换持久化 Checkpointer。

## 5. 预算与排错

| 参数 | 默认 | 作用 |
|---|---:|---|
| `memory_limit` | 64 MB | 每线程 QuickJS 堆内存 |
| `timeout` | 5 秒 | 每次 eval 执行时间 |
| `tool_name` | `"eval"` | 暴露给模型的工具名 |
| `capture_console` | `True` | 是否返回 console 输出 |
| `max_result_chars` | `4000` | 截断结果、错误与标准输出 |
| `ptc` | `None` | PTC 白名单；省略即关闭 |
| `max_ptc_calls` | `256` | 每次 eval 的 PTC 调用数 |
| `subagents` | `True` | 有子 Agent 时是否暴露 `task()` |
| `mode` | `"thread"` | 状态保留范围 |
| `max_snapshot_bytes` | `None` | 快照大小，默认跟随内存上限 |

`max_ptc_calls=256` 是运行时上限，不是建议用满；订单实验只有七个输入，主动收紧到 16。真实应用按最大合法批次设置，并让工具自身继续做鉴权、配额与参数校验。

结果超过 `max_result_chars` 会截断。不要简单调大上限，把大批原始数据塞回上下文；先在 JS 里筛选、计数或分组，再返回支撑最终答案的最小结果。

| 现象 | 先查 | 处理 |
|---|---|---|
| Agent 从不调用 `eval` | 模型是否支持 Tool Calling；任务是否真需批处理 | 换模型，并明确要求用 eval 完成循环/聚合 |
| JS 中没有目标工具 | `ptc` 用名称还是对象 | 名称匹配 Agent 工具集；对象直接放列表 |
| 提示 `tools.lookup_order` 不存在 | 是否用了转换后函数名 | 改 `tools.lookupOrder` |
| 工具输入校验失败 | 参数是否仍按 Python Schema | 保留 `order_id`，不要改 `orderId` |
| 批处理跑到一半停 | 是否触发 timeout 或 max_ptc_calls | 缩小批次，检查终止条件，按合法上限调整 |
| 结果缺少后半段 | 是否触发 max_result_chars | 解释器内先聚合，不返回全部原始记录 |
| 跨轮读函数报恢复错误 | Snapshot 是否含不可序列化对象 | 跨轮只保存数据，下轮重新定义函数 |
| 恢复旧快照后外部数据没回滚 | 是否把 Snapshot 当事务 | 为副作用工具设计幂等键、补偿或真事务 |

Event Streaming 仍可观察主 Agent 消息与普通工具调用；PTC 中间步骤由 PTC runtime/tool proxy 执行，不要假设前端会把每个 `tools.*` 调用渲染成普通工具卡片。需要审计时，同时记录 eval 输入、PTC 工具自身日志与最终结果摘要，并对敏感参数脱敏。

## 6. 收紧安全边界

QuickJS 默认没有文件、网络与 Shell；一旦把工具接入 PTC，**工具能做什么，解释器代码就能做什么**。

设计白名单先问四个问题：

1. 任务是否真的需要该工具？
2. 工具能否读秘密、访问任意路径或调用任意 URL？
3. 工具是否会花钱、改数据或触发外部通知？
4. 单次 eval 最多允许调用多少次？

优先暴露窄工具（「按 ID 读一个订单」），不要暴露「执行任意 SQL」或「向任意 URL 发请求」。工具内部仍要校验调用者、资源范围与参数；**PTC 白名单不能替代业务鉴权**。

### QuickJS 不是宿主内存隔离边界

解释器代码运行在嵌入式 QuickJS Context，不是独立 VM 或进程。它能限制默认能力与资源，却不是完整的宿主内存隔离方案。

面对不可信或半可信代码，把整个 Agent 放进隔离 Worker 或容器，并继续收紧 PTC 白名单。需要 Shell、装包或系统级测试用 Sandbox；需要人工确认高风险动作保留 HITL；文件访问范围继续由权限规则控制。这些机制负责不同边界，不能互相替代。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_51-framework-ptc-safety-guardrails.png

<p>图 4：PTC 的四层防护</p>

### 什么时候不该用 Interpreter

- 只有一次查询：普通工具调用更容易观察与审批；
- 任务主要依赖自然语言判断：让模型直接推理更清楚；
- 需要操作系统能力：用 Sandbox，不要用宽泛 PTC 工具绕过边界；
- 每个写操作都要单独审批：保留普通工具路径；
- 需要多个独立角色完成完整推理：用 Dynamic Subagents。

## 7. 检查订单实验

提交 `A-100` 到 `A-105`，再加不存在的 `A-999`，要求用 `eval` 并行查询、筛选、排序、汇总。模型措辞不固定，不要匹配整段答案，检查五件事：

1. Agent 调用了 `eval`，而不是逐个发七次普通调用；
2. JS 用 `Promise.all` 或等价方式覆盖全部订单号；
3. `A-999` 归入缺失项，不会让整批失败；
4. 风险订单按金额降序返回，总金额由代码计算；
5. 最终模型只收到整理后的对象。

代表性结果：

```text
风险订单：A-104、A-101、A-102、A-105
风险订单总金额：5320
未找到：A-999
```

如果 Trace 显示没用 `eval`，先查模型 Tool Calling 能力与系统提示词；不要为了「让示例看起来成功」删掉七个输入——批处理正是要验证的行为。

## 8. 本篇验收

- [ ] 能说出普通 Tool Calling 在批量任务上的四个问题；
- [ ] 能判断 Tool Calling / Interpreter / PTC / Sandbox / 动态子 Agent 的适用边界；
- [ ] 知道 `eval` 返回最后一个表达式的值；
- [ ] 记得 QuickJS 默认没有文件、网络、Shell、时间；
- [ ] 会用 PTC 对象与名称两种白名单，并知道函数名转 camelCase、参数不转；
- [ ] 理解 PTC 不逐次走 `interrupt_on`，高副作用工具不应入白名单；
- [ ] 能按 call/turn/thread 选择状态范围，知道 Snapshot 只存可序列化数据；
- [ ] 会收紧 timeout、memory、max_ptc_calls 与结果上限；
- [ ] 记住 QuickJS 不是沙箱或宿主内存隔离。

> **补记**：Interpreter 真正替代的是「模型逐轮做确定性编排」，不是「模型推理」。因此判断标准很简单：**能用代码确定性完成的循环、过滤、聚合就交给 eval；需要判断、审批或操作系统能力的留在外面**。最危险的一步是把高副作用工具放进 PTC 白名单——那条路径不逐次走人工审批，白名单本身就是权限边界。

## 写在最后

下一篇[动态子 Agent](17-动态子Agent)把 QuickJS 里的 `tools.*` 扩展成 `task()`：用代码动态创建并编排多个完整子 Agent，而不是预声明固定角色。
