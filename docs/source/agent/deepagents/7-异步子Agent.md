# 异步子 Agent：不阻塞主线的委派

## 1. 同步委派的瓶颈

同步子 Agent 通过 `task` 委派后，主 Agent **等待**子 Agent 跑完。子任务要几分钟甚至几十分钟时，用户在对话里只看到「转圈」——既不能继续聊，也不能插话调整方向。

两类场景体验最差：**长程任务**（深度调研、大规模代码迁移、批量数据处理）与**可交互任务**（跑到一半想补充「换个数据源」「加上 2024 年数据」）。更糟的是，`task()` 期间主 Agent 自己也被阻塞：子 Agent 完成前，用户无法和主 Agent 聊别的话题。

异步子 Agent 解决两件事：**不阻塞主线对话**、**支持中途控制**。

## 2. 同步 vs 异步：六个维度

| 维度 | 同步子 Agent | 异步子 Agent |
|---|---|---|
| 执行模型 | 阻塞，等完成才能继续 | 非阻塞，立即返回任务 ID |
| 并发性 | 可并行触发，但主 Agent 被整批阻塞 | 完全并行，主 Agent 全程不阻塞 |
| 中途追加指令 | 不支持 | `update_async_task` 注入新指令 |
| 取消 | 不支持 | `cancel_async_task` 请求取消 |
| 状态性 | 无状态，每次调用相互独立 | 有状态，子 Agent 拥有自己的会话（thread），历史持续累积 |
| 典型场景 | 毫秒到秒级的快速委派 | 分钟以上的研究、编码、迁移等长程任务 |

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_15-comparison-sync-vs-async.png)

<p>图 1：同步与异步子 Agent 对比</p>

判定法则：**子任务能在 5 秒内完成用同步；可能跑数分钟以上、且过程需要交互管理，就上异步。**

## 3. 先分清协议、服务与追踪

异步子 Agent 需要服务端承接后台任务，涉及四个不同层次：

| 名称 | 职责 |
|---|---|
| Agent Protocol | 调用 Agent 的 API 规范：创建会话、启动运行、查询状态、取消任务 |
| Agent Server | 实现这些接口的运行服务，加载 Agent 代码、调度执行、管理会话与结果 |
| LangSmith Deployment | 部署与运行 Agent Server 的平台能力；也可自托管兼容服务 |
| LangSmith Observability | 采集与查看 trace，分析模型调用、工具执行、耗时与错误 |

主 Agent 决定任务怎么拆、交给谁、如何整合；Agent Server 负责承接与执行。开启 LangSmith 追踪会记录过程，但**不会把本地 Agent 自动变成可接收任务的服务**。

服务端的 **thread 是保存消息与状态的会话，不是操作系统线程**；**run 是该会话上的一次执行**。主 Agent 调用 `start_async_task` 后，中间件通过 SDK 创建子任务的 thread 与 run，返回任务 ID，服务端继续执行子 Agent。

## 4. 声明异步子 Agent

`AsyncSubAgent` 是一份配置：`graph_id` 标识服务端已注册的 graph/assistant，`url` 指定服务地址。多个子 Agent 可共用一个 Agent Server；与主 Agent 同部署时可省略 `url`，走进程内 ASGI 传输。

```python
from deepagents import AsyncSubAgent, create_deep_agent

async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="深度研究 Agent，用于多次搜索 + 信息综合的调研任务",
        graph_id="researcher",
        # 不传 url → ASGI 进程内传输（与主 Agent 同部署）
    ),
    AsyncSubAgent(
        name="coder",
        description="编码 Agent，用于代码生成、改写与代码评审",
        graph_id="coder",
        # url="https://coder-deployment.langsmith.dev"   # 远程 HTTP 传输
    ),
]

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=async_subagents,
)
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 唯一标识，主 Agent 启动任务时指定 |
| `description` | 是 | 能力描述，主 Agent 据此决定何时委派 |
| `graph_id` | 是 | 服务上的 graph/assistant ID；LangGraph 部署要在 `langgraph.json` 注册 |
| `url` | 否 | 不填走进程内 ASGI；填了走远程 HTTP |
| `headers` | 否 | 远程模式下的自定义 HTTP 头，用于自托管鉴权 |

> **前置提醒**：Async Subagent 是 `deepagents>=0.5.0` 的**预览特性**，API 可能变化。与前几章直接在 Python 进程 `agent.invoke()` 不同，本篇示例需要运行在支持 Agent Protocol 的 LangGraph 服务环境中。

## 5. 主 Agent 的 5 把遥控器

声明 `AsyncSubAgent` 后，`AsyncSubAgentMiddleware` 自动给主 Agent 注入 5 个工具：

| 工具 | 作用 | 返回 |
|---|---|---|
| `start_async_task` | 启动后台任务 | 任务 ID（立即） |
| `check_async_task` | 查询状态与结果 | 状态 + 结果（若完成） |
| `update_async_task` | 给运行中任务追加指令 | 确认 + 更新后状态 |
| `cancel_async_task` | 终止运行中任务 | 确认 |
| `list_async_tasks` | 列出所有任务及实时状态 | 任务总览 |

一次标准对话：

```text
用户：帮我深入调研 LangGraph 多 Agent 架构。
主 Agent → start_async_task(description="调研 LangGraph 多 Agent 架构", subagent_type="researcher")
           → 返回 task_id: abc-123
主 Agent ←「已派 researcher 在后台开干，任务 ID：abc-123，你可以继续问别的，或随时查进度。」

用户：先帮我把这段 Python 代码格式化一下。
主 Agent ←（直接处理，researcher 仍在后台跑）

用户：刚才那个调研有进展吗？
主 Agent → check_async_task("abc-123") → status: running
主 Agent ←「还在跑，已搜 4 个关键词，预计还要几分钟。」

用户：补一下：重点关注 supervisor / network / hierarchical 三种拓扑。
主 Agent → update_async_task("abc-123", "重点关注三种拓扑…") → 已注入

用户：算了，先停一下。
主 Agent → cancel_async_task("abc-123") → cancelled
```

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_16-flowchart-async-lifecycle.png)

<p>图 2：异步任务生命周期</p>

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_17-sequence-async-protocol.png)

<p>图 3：用户、主 Agent 与 Agent Protocol 服务的时序</p>

**五个工具底层做了什么**：

- **start（launch）**：在服务上新建 thread、启动 run，把任务描述作为输入，返回 thread ID 作为 task ID；主 Agent 拿到 ID 就回到对话循环，**不轮询等待**；
- **check**：读 run 当前状态；完成后再读 thread state 拿最终输出；仍在跑就告诉用户「运行中」；
- **update**：在**同一 thread** 上以 `interrupt` 多任务策略发起新 run——前一次被打断，子 Agent 带完整历史 + 新指令重启，任务 ID 不变；
- **cancel**：调用 `runs.cancel()` 终止远程 run，本地标记 `cancelled`；
- **list**：遍历跟踪中的任务；已结束的从缓存返回，未结束的并发拉取实时状态。

## 6. 为什么要单开 `async_tasks` 通道

主 Agent 的 graph state 里有一个专门的 `async_tasks` 通道，独立于消息历史，保存每个任务的 task ID、子 Agent 名、thread ID、run ID、状态与时间戳。

为什么不放工具消息里？因为上下文逼近上限时会**自动压缩对话历史**。如果任务 ID 只活在某条 `ToolMessage` 里，压缩后就丢了——主 Agent 会「忘了」自己派出去的所有任务，再也无法 check 或 cancel。搬到独立 channel 后，消息历史可以放心压缩，主 Agent 永远能通过 `list_async_tasks` 找回全部任务。

> 这是 Deep Agents 一贯的设计哲学：**会被截断的放消息历史，必须长存的进 state channel**——与虚拟文件系统、TodoList 同一套思路。

## 7. ASGI 与 HTTP 两种传输

```text
HTTP 传输：SDK → 网络连接 → Uvicorn → Agent Server 应用
ASGI 传输：SDK → 进程内 ASGI 适配器 → Agent Server 应用
```

**ASGI（同部署，推荐起手式）**：主 Agent 与子 Agent 注册到同一份 `langgraph.json`，用 `langgraph dev` 启动；`AsyncSubAgent` 省略 `url`，只用 `graph_id`。优势是零网络延迟、零额外鉴权，子 Agent 仍有独立 thread。注意：ASGI 传输需要主 Agent 通过 `ainvoke()` 等异步入口执行，不支持同步 `invoke()`。

**本地最小验证方案**（单部署 + ASGI，可直接跑）：

```text
async-subagent-demo/
├── .env
├── langgraph.json
├── requirements.txt
├── run_demo.py
└── graphs/
    ├── researcher.py
    └── supervisor.py
```

```bash
uv venv && source .venv/bin/activate
uv pip install -U "langgraph-cli[inmem]"
uv pip install -r requirements.txt   # deepagents>=0.5.0 / langgraph / langgraph-sdk / langchain-openai
```

`langgraph.json` 的关键点：`dependencies` 指向 `./`；`supervisor` 与 `researcher` 同时注册，`AsyncSubAgent` 才能用 ASGI 直接调用：

```json
{
  "dependencies": ["./"],
  "graphs": {
    "supervisor": "./graphs/supervisor.py:graph",
    "researcher": "./graphs/researcher.py:graph"
  },
  "env": "./.env"
}
```

为了让异步效果稳定可见，`researcher` 可以故意 `sleep 8 秒` 再返回：

```python
import asyncio
from langgraph.graph import END, START, MessagesState, StateGraph

async def slow_research(state: MessagesState):
    last_human = state["messages"][-1].content if state["messages"] else "No task."
    await asyncio.sleep(8)
    return {"messages": [{"role": "ai", "content": f"[researcher finished after 8s]\ntask: {last_human}"}]}

builder = StateGraph(MessagesState)
builder.add_node("slow_research", slow_research)
builder.add_edge(START, "slow_research")
builder.add_edge("slow_research", END)
graph = builder.compile()
```

`supervisor.py` 的关键是三条：`graph_id="researcher"` 与配置键名一致；**不要写 url**（才会走 ASGI）；提示词明确「启动后立即把 task_id 返回用户，不要立刻轮询」：

```python
graph = create_deep_agent(
    model=os.environ.get("MODEL_NAME", "openai:gpt-4.1-mini"),
    system_prompt=(
        "You are a supervisor agent for an async-subagent demo. "
        "When the user asks for a long-running research task, delegate to researcher immediately. "
        "After calling start_async_task, return the task_id to the user and stop. "
        "Do not call check_async_task unless the user explicitly asks for progress. "
        "If the user asks to revise the background task, call update_async_task."
    ),
    subagents=[AsyncSubAgent(name="researcher", description="长程后台研究", graph_id="researcher")],
)
```

启动与验证：

```bash
langgraph dev --n-jobs-per-worker 4     # 至少容纳 1 个 Supervisor + 1 个 Researcher
```

SDK 脚本在同一 thread 上连续做三件事：启动后台任务、立刻追问进度（验证未阻塞）、追加约束（验证 update）：

```python
import asyncio
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")

async def main():
    thread = await client.threads.create()
    tid = thread["thread_id"]
    for q in [
        "请把这个任务交给 researcher 异步处理：总结 async subagent 的关键行为。",
        "刚才那个后台任务现在进展如何？",
        "补充约束：完成时请把答案写成 3 条 bullet。",
    ]:
        r = await client.runs.wait(tid, "supervisor", input={"messages": [{"role": "user", "content": q}]})
        print(r)

asyncio.run(main())
```

应看到：第一次很快返回任务 ID，而不是卡 8 秒；第二次大概率 `running`；第三次更新已有任务；几秒后再查变为 `success` 并带最终结果。

**HTTP 传输**：加 `url` 即切换远程：

```python
AsyncSubAgent(
    name="researcher",
    description="Research Agent",
    graph_id="researcher",
    url="https://my-research-deployment.langsmith.dev",
)
```

LangGraph 部署的鉴权由 SDK 自动处理（读 `LANGSMITH_API_KEY` 或 `LANGGRAPH_API_KEY`）；自托管服务用 `headers` 传。适合子 Agent 需要独立扩缩容、不同资源画像（GPU/大内存）、或由另一团队维护的场景。

## 8. 三种部署拓扑

| 拓扑 | 形态 | 推荐场景 |
|---|---|---|
| 单部署 | 所有 Agent 同部署，全 ASGI | 绝大多数项目的起点：好运维、零网络延迟 |
| 拆分部署 | 主 Agent 一台、子 Agent 一台，全 HTTP | 资源画像或扩缩容策略显著不同 |
| 混合 | 一部分 ASGI、一部分 HTTP | 多数同部署，少数特殊子 Agent 单独扩 |

**起手式建议**：先用单部署 + ASGI，等遇到具体扩缩容或团队边界问题再拆。

## 9. 最佳实践与排查

**Worker Pool 要够**：每个活跃运行占一个槽位；主 Agent 同时跑 3 个子 Agent 至少需要 4 个槽位。不够时新任务排队，表现为 `start_async_task` 长时间不返回，或拿到 ID 后 `check` 长时间无实质进展。

```bash
langgraph dev --n-jobs-per-worker 10
```

**描述要具体**：「深度网络调研，需要多次搜索 + 信息综合时使用」好于「帮你处理事情」。

**用 thread ID 串联追踪**：每次异步运行都是一次普通 LangGraph run；启用 LangSmith 后，主 Agent 的 trace 显示 launch/check/update/cancel/list，子 Agent 是另一条 trace，用 thread ID（即 task ID）把两边对上。

| 现象 | 优先检查 | 处理 |
|---|---|---|
| 找不到 Agent | `graph_id` 与 `langgraph.json` 注册名是否一致 | 注意大小写与下划线 |
| 远程 HTTP 调用失败 | `url`、`headers`、`LANGSMITH_API_KEY`/`LANGGRAPH_API_KEY` | 自托管服务把鉴权头放进 `headers` |
| 本地能跑、拆分后失败 | 子 Agent 服务是否兼容 Agent Protocol | 先用 SDK 直接创建 thread/run 验证 |
| 一直 `running` | worker 数、外部工具超时、子 Agent 是否卡在长调用 | 提高槽位，给外部 API 设超时 |
| cancel 后状态没立刻变 | 服务端取消异步生效 | 再 check 或 list 确认最终状态 |
| 查不到之前的任务 | thread/checkpoint 是否持久化 | 主 Agent 配 Checkpointer，`async_tasks` 才能恢复 |
| trace 对不上 | task/thread/run ID 是否完整 | 保留完整 task ID，用 thread ID 串联 |

另外三类模型行为问题：**启动后立刻轮询**（在提示词里强化「交还控制权，不要主动 check」）、**引用过时状态**（要求回答进度前必须先 check/list）、**截断 task ID**（要求始终使用完整 ID，或换更听话的模型）。

## 10. 本篇验收

- [ ] 能说出同步委派的两个瓶颈与异步的解法；
- [ ] 记住 5 个工具的分工与 launch 的实际行为；
- [ ] 理解 thread/run 与「thread 不是操作系统线程」；
- [ ] 能解释 `async_tasks` 独立通道与上下文压缩的关系；
- [ ] 会在 ASGI 与 HTTP 之间选择，并知道三种部署拓扑；
- [ ] 能按最小示例注册 `langgraph.json` 并跑通验证；
- [ ] 知道 worker pool、完整 task ID、实时 check 三个高频坑。

> **补记**：异步子 Agent 的关键不是「开个后台任务」，而是**把任务生命周期从对话历史里解耦**：任务 ID、状态、时间戳放进独立 state channel，消息被压缩也不会丢。工程上最该先做对的三件事：`graph_id` 与注册名一致、worker 槽位留够、提示词要求「派完就交还控制权且回答进度前必须实时 check」。另外它是 preview 特性，升级依赖前先看变更说明。

## 写在最后

下一篇[Skills 能力包](8-Skills能力包)讲如何把领域知识、操作流程与脚本封装成可复用技能，让 Agent 按需加载而不是一次塞满上下文。
