# Human-in-the-Loop：为敏感操作加审批

## 1. 为什么需要 HITL

Agent 的自主性是一把双刃剑：好处是能独立完成复杂任务；风险是可能执行危险操作——删错文件、发错邮件、调用昂贵 API。实际项目里，完全自主与完全人工之间需要一个**可控的中间地带**：

- 删除文件前，先让用户确认；
- 发送邮件前，让用户检查收件人与内容；
- 修改生产配置前，必须获得审批。

这就是 **Human-in-the-Loop（HITL）**：Agent 在执行特定操作前**暂停**，等待人类审批、修改、拒绝或直接响应，然后继续。

## 2. `interrupt_on` 配置

Deep Agents 用 `interrupt_on` 指定哪些工具需要审批。设置后会在默认中间件栈加入 `HumanInTheLoopMiddleware`；若运行在工具返回前被中断或取消，`PatchToolCallsMiddleware` 会自动修复消息历史。

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model=model,
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "read_file": False,    # 无需中断
        "send_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=MemorySaver(),   # 必须配置
)
```

`interrupt_on` 是「工具调用审批」的便捷入口，不是 HITL 的能力边界。暂停点不对应某个工具时（发布最终答复前审稿、进入下一阶段前确认、跨多个工具统一策略），可以在自定义 LangChain Middleware 中直接调用底层 `interrupt()`。

**三种配置值**：

| 值 | 含义 |
|---|---|
| `True` | 启用中断，允许所有决策（approve/edit/reject/respond） |
| `False` | 不中断，直接执行 |
| `{"allowed_decisions": [...]}` | 启用中断，只允许指定决策 |

## 3. 四种决策

| 决策 | 含义 | 场景 |
|---|---|---|
| `approve` | 批准，使用 Agent 原始参数 | 确认删除这个文件 |
| `edit` | 修改参数后执行 | 收件人改一下再发 |
| `reject` | 跳过此次调用，把拒绝原因反馈给 Agent | 不要删除，取消 |
| `respond` | 不执行工具，把人的 `message` 作为成功的合成工具结果 | `ask_user` 这类询问工具 |

> **拒绝副作用工具用 `reject`，不要用 `respond`**。`respond` 的内容会被模型当作一次成功的 `ToolMessage`；删除、发送、部署这类工具应该用 `reject` 明确告诉 Agent 没有执行。

三条规则：**不同意执行** → `reject` + 说明原因与下一步；**同意但改参数** → `edit` 只改必要参数；**工具本来就是问人** → `respond` 让回答成为工具结果。

## 4. 条件中断

默认只要工具名在 `interrupt_on` 里，每次调用都暂停。只想拦截某些参数组合时加 `when` 谓词（需要 `langchain>=1.3.3`）：

```python
from langchain.agents.middleware import ToolCallRequest

def writes_outside_workspace(request: ToolCallRequest) -> bool:
    path = request.tool_call["args"].get("file_path", "")
    return not path.startswith("/workspace/")

agent = create_deep_agent(
    model=model,
    interrupt_on={
        "write_file": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": writes_outside_workspace,
        },
    },
    checkpointer=MemorySaver(),
)
```

`when` 返回 `False` 时该调用不进入中断批次，审批界面只展示真正需要人工决策的动作。

## 5. 中断与恢复

流程：**Agent 运行到敏感工具 → 暂停并返回中断信息 → 用户检查并决策 → 用相同 `thread_id` 恢复。**

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_17-flowchart-hitl-flow.png

<p>图 1：HITL 执行流程</p>

```python
import uuid
from langgraph.types import Command

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "删除 temp.txt 文件"}]},
    config=config,
    version="v2",
)

if result.interrupts:
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    for action in action_requests:
        review_config = config_map[action["name"]]
        args = action.get("arguments", action.get("args", {}))
        print(action["name"], args, review_config["allowed_decisions"])

    decisions = [{"type": "approve"}]

    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,      # 同一个 thread_id
        version="v2",
    )

print(result.value["messages"][-1].content)
```

**关键要求**：

- **必须配 Checkpointer**：HITL 依赖状态持久化，没有就无法恢复；
- **必须用相同 `thread_id`**：中断与恢复在同一线程；
- **返回格式**：`version="v2"` 用 `result.interrupts`/`result.value` 读取；默认 `invoke()` 也支持，中断在返回字典的 `__interrupt__` 里，v2 不是必要条件；
- **决策数量与顺序要匹配** `action_requests`。

恢复时 LangGraph **重新进入发生中断的节点**，而不是恢复 Python 调用栈。`interrupt_on` 的内置审批发生在独立的 `after_model` 节点，模型生成的工具请求已进入 State，恢复审批会复用该请求。若在同一节点内先调用模型或执行外部写操作、再调用 `interrupt()`，中断前的代码可能再次执行——Checkpointer 不会自动保证幂等。

> 字段差异：Deep Agents 文档常见 `action_request["args"]`，LangChain 标准 `HumanInTheLoopMiddleware` 示例用 `["arguments"]`。兼容两种入口可先读 `arguments` 再回退 `args`；但恢复 `edit` 决策时，`edited_action` 仍使用 `args`。

## 6. 拒绝、响应与编辑

**拒绝要写清反馈**：

```python
decisions = [{
    "type": "reject",
    "message": "用户拒绝删除该文件。不要再次尝试删除，请询问是否改为归档文件。",
}]
```

不传 `message` 时，默认反馈会告诉模型工具没有执行、不要重复调用。

**`respond` 只用于「问用户」的工具**：`ask_user` 工具本身不执行，人的回答直接作为成功结果交还：

```python
@tool
def ask_user(question: str) -> str:
    """向用户提问；真实回答由 HITL 的 respond 决策提供。"""
    return "等待用户回答"

decisions = [{"type": "respond", "message": "使用季度维度，并排除测试数据。"}]
```

**编辑参数**：决策类型为 `edit` 时修改参数再执行：

```python
decisions = [{
    "type": "edit",
    "edited_action": {
        "name": action_request["name"],   # 必须包含工具名
        "args": {"to": "team@example.com", "subject": "通知", "body": "..."},
    }
}]
```

编辑尽量保守（改收件人、路径、SQL 条件），大幅改写可能让模型重新评估计划、重复调用或走向意外动作。

**批量调用**：多个需审批的工具会打包成一个中断，按顺序为每个工具提供决策：

```python
decisions = [
    {"type": "approve"},
    {"type": "reject", "message": "用户拒绝发送邮件。不要重试这次发送动作。"},
]
```

## 7. 子 Agent 独立配置

子 Agent 可以有独立的 `interrupt_on`，覆盖主 Agent：

```python
agent = create_deep_agent(
    model=model,
    tools=[delete_file, read_file],
    interrupt_on={"delete_file": True, "read_file": False},
    subagents=[{
        "name": "file-manager",
        "description": "管理文件操作",
        "system_prompt": "你是文件管理助手。",
        "tools": [delete_file, read_file],
        "interrupt_on": {"delete_file": True, "read_file": True},  # 更严格
    }],
    checkpointer=checkpointer,
)
```

主 Agent 可以信任，但子 Agent 可能操作更敏感的数据，需要更严格的审批。

## 8. 按风险分层

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_18-infographic-risk-levels.png

<p>图 2：按风险等级分层配置</p>

```python
interrupt_on = {
    # 高风险：审批 + 修改 + 拒绝，不开放 respond
    "delete": {"allowed_decisions": ["approve", "edit", "reject"]},
    "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
    "execute_sql": {"allowed_decisions": ["approve", "edit", "reject"]},
    "deploy_to_production": {"allowed_decisions": ["approve", "edit", "reject"]},

    # 中风险：审批或拒绝，不允许改参数
    "edit_file": {"allowed_decisions": ["approve", "reject"]},
    "call_external_api": {"allowed_decisions": ["approve", "reject"]},

    # 低风险：不需要中断
    "read_file": False, "ls": False, "grep": False, "glob": False,

    # 人工输入型
    "ask_user": {"allowed_decisions": ["respond"]},
}
```

| 风险 | 工具类型 | 配置 | 理由 |
|---|---|---|---|
| 高 | 删除、覆盖写入、发送、部署 | approve/edit/reject | 不可逆或影响外部系统，避免 respond 被当成成功 |
| 中 | 写入、外部调用 | approve/reject | 可审批，无需改参数 |
| 低 | 读取、搜索、列表 | False | 只读无副作用 |
| 输入型 | 询问偏好、补信息 | respond | 人的回答就是工具结果 |

## 9. 文件系统权限中断

内置文件工具也能被权限规则触发中断（需要 `deepagents>=0.6.8`）：

```python
agent = create_deep_agent(
    model=model,
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt"),
    ],
    checkpointer=MemorySaver(),
)
```

`write_file`、`edit_file`、`delete` 命中规则时，抛出与普通工具审批相同格式的 HITL 中断；恢复方式一致（检查 `action_requests` → `Command(resume=...)`）。文件权限中断会与 `interrupt_on` 合并，一次人工审查可同时覆盖自定义工具与受保护路径。

> v0.7 中 `write_file` 会完整覆盖，`delete` 可递归删除目录，因此 `operations=["write"]` 保护的不只是新建/编辑，也包括覆盖与删除；包含受保护后代的目录删除会整体拒绝，不会只删一部分。真实文件系统敏感路径优先用路径权限，`interrupt_on` 更适合自定义工具或统一审批策略。

## 10. 底层机制：`interrupt()`

`interrupt_on` 的底层是 LangGraph 的**中断原语**。在工具或节点中直接调用 `interrupt()` 可以实现更灵活的审批：

```python
from langgraph.types import interrupt

@tool
def request_approval(action_description: str) -> str:
    """请求人工审批。"""
    approval = interrupt({
        "type": "approval_request",
        "action": action_description,
        "message": f"请审批：{action_description}",
    })
    return "已获批准" if approval.get("approved") else f"被拒绝：{approval.get('reason')}"
```

恢复：

```python
agent.invoke(Command(resume={"approved": True}), config=config)
agent.invoke(Command(resume={"approved": False, "reason": "时机不对"}), config=config)
```

**在自定义 Middleware 中**：Node-style Hook 会成为独立图节点，可直接调用 `interrupt()`。官方 `HumanInTheLoopMiddleware` 就是在 `after_model` 中检查工具调用并暂停。例如「模型生成最终草稿后、交付用户前统一审稿」这类跨工具策略：

```python
class DraftApprovalMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        decision = interrupt({"type": "draft_review", "draft": last.content})
        if decision.get("approved"):
            return None
        return {"messages": [AIMessage(content=f"草稿未发布：{decision.get('reason')}")]}
```

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_29-flowchart-interrupt-state-resume.png

<p>图 3：interrupt 的保存与恢复</p>

**运行时视角**：`interrupt(value)` 抛出特殊异常 → 运行时包装成 `Interrupt`（含审查数据与稳定 id）→ Checkpointer 保存线程状态、下一步节点与待恢复中断 → 调用方拿到 `interrupts` → 人类返回 `Command(resume=...)` → 运行时按同一 `thread_id` 找回 checkpoint，把恢复值送回对应 `interrupt()`。所以 HITL 的关键不是「弹确认框」，而是**任意时间跨度后仍能安全恢复**：几秒、几小时、甚至换进程，只要 Checkpointer 与 `thread_id` 还在。

**五条使用规则**：

1. **不要用裸 try/except 包裹 `interrupt()`**：它靠异常暂停，裸 `except` 会吞掉异常；只捕获具体异常类型。
2. **`interrupt()` 之前的副作用必须幂等**：恢复时节点从头重放，`db.create_log()` 会重复执行；用 upsert，或把副作用放到 `interrupt()` 之后。
3. **不要动态改变 `interrupt()` 的调用顺序**：同一节点内多个中断按索引匹配，条件跳过会导致错位。
4. **并行中断用 ID 映射**：`Command(resume={intr.id: answer ...})`，避免界面排序或子图重排导致错配。
5. **resume 后从节点开头重放**：`interrupt()` 会用已有恢复值直接返回，不再暂停；节点代码要能承受中断前逻辑再跑一次。

**检查暂停位置**：本地 `graph.get_state(config)` 可看 `values`、`next`、`interrupts`、`checkpoint`；平台/SDK 用 `client.threads.get_state(thread_id)`。`interrupts` 还在说明缺 resume；`thread_id` 不对会看到另一个线程的状态。

**扩展模式**：

- **输入验证**：不要在同一节点 `while True` 反复 `interrupt()`（恢复会重放前几轮）；每次节点只调用一次 `interrupt()`，把下一问写回状态，用条件边决定是否回到同一节点；
- **静态中断**：`compile(interrupt_before=[...], interrupt_after=[...])` 适合开发调试，生产人机协作仍用动态 `interrupt()`；
- **何时直接使用底层**：工具审批用 `interrupt_on`；按参数审批用 `interrupt_on + when`；模型前后统一审查用 Node-style Middleware + `interrupt()`；节点内收集缺失信息、多轮表单、并行汇总用底层 `interrupt()`；调试图路径用静态中断。

> Wrap-style Hook 虽然也能调用 `interrupt()`，但运行在 model/tools 节点内部，`handler` 可能被跳过、执行一次或因重试执行多次；恢复时整个节点从头重放，模型或工具调用可能再次发生。重试、缓存、转换适合 Wrap-style，**人工中断边界应放 Node-style**。

## 11. 本篇验收

- [ ] 能说出 HITL 要解决的三个典型风险；
- [ ] 会配 `interrupt_on` 的三种值与四种决策；
- [ ] 记住「拒绝副作用用 reject，respond 只给询问工具」；
- [ ] 能用 `when` 做条件中断；
- [ ] 能完成「暂停 → 决策 → 同 thread 恢复」全流程；
- [ ] 知道恢复是重进节点，中断前副作用必须幂等；
- [ ] 会给子 Agent 配更严格的审批；
- [ ] 能按风险分层配置工具；
- [ ] 会用 `FilesystemPermission(mode="interrupt")` 保护敏感路径；
- [ ] 理解 `interrupt()` 的异常暂停、Checkpointer 保存与五条规则。

> **补记**：HITL 的核心不是「多一步确认」，而是**把执行权在关键时刻交还给人，同时保证之后能可靠恢复**。工程上最该记住三条：Checkpointer + 同一 `thread_id` 是恢复前提；恢复会重放节点，`interrupt()` 前的副作用必须幂等；`reject` 与 `respond` 语义完全不同——前者表示「没执行」，后者表示「人替你执行了」。

## 写在最后

下一篇[沙箱执行](11-沙箱执行)让 Agent 在隔离环境里运行代码：执行后端、隔离边界与文件平面，以及生产环境的选型。
