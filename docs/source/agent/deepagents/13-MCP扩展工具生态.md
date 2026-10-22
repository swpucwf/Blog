# MCP：用标准协议扩展工具生态

## 1. MCP 在 Deep Agents 中的位置

自定义工具适合接入少量、稳定的能力；当 Agent 要连越来越多数据库、文件系统与外部服务时，逐个维护适配代码会迅速失控。**Model Context Protocol（MCP）** 用统一边界描述 Agent 可用的工具与上下文。

| 层次 | 组件 | 负责什么 |
|---|---|---|
| Agent Harness | Deep Agents | 文件系统、上下文、子 Agent、工具调用循环 |
| Agent 工具接口 | LangChain Tool | 让模型看到工具名、描述、参数 Schema 并执行 |
| 协议适配 | `langchain-mcp-adapters` | 把 MCP 工具转成 LangChain Tool |
| MCP Client | `MultiServerMCPClient` | 连接一个或多个 Server，管理发现与调用会话 |
| MCP Server | 本地进程或远程服务 | 执行数据库、API、文件或业务操作 |

```text
用户请求
  -> Deep Agent 决定调用工具
  -> LangChain Tool 校验参数
  -> MCP Client 建立 Session
  -> MCP Server 执行业务逻辑
  -> 结果转换成 ToolMessage
  -> Deep Agent 继续推理并回答
```

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_36-framework-mcp-integration-layers.png)

<p>图 1：MCP 接入的分层架构</p>

`create_deep_agent()` **不直接接收 MCP 配置，也不管理 Session 生命周期**。应用先通过客户端加载工具，再把 LangChain Tool 列表传给 `tools=`：

```python
tools = await client.get_tools()
agent = create_deep_agent(model=model, tools=tools)
```

这些工具与文件系统、子 Agent 等 Harness 能力一起提供给模型。MCP Server 不会因此自动获得 Deep Agents 的 State、Store 或 Backend——它仍是边界外的独立进程或服务。

### 1.1 Tools、Resources 与 Prompts

| MCP 能力 | 转换结果 | 直接给 Agent？ | 用途 |
|---|---|---|---|
| Tools | `BaseTool` / `StructuredTool` | 是 | 查数据库、调 API、执行业务 |
| Resources | `Blob` | 否 | 文件/记录/二进制，由应用决定如何注入 |
| Prompts | 消息列表 | 否 | 可复用提示模板，由应用决定放哪段对话 |

后两者不会因为 `get_tools()` 自动进入模型上下文。

## 2. 创建本地 MCP Server

```bash
mkdir deepagents-mcp-demo && cd deepagents-mcp-demo
uv init --bare --python 3.11
uv add --upgrade "deepagents>=0.7,<0.8" "langchain-mcp-adapters>=0.3,<0.4" "mcp>=1.28,<2" langchain-openai
```

`math_server.py`：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Chapter 12 Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers exactly."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers exactly."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`@mcp.tool()` 把普通函数转换成 MCP Tool，三处信息直接决定模型如何使用：

| Python 定义 | MCP 字段 | 作用 |
|---|---|---|
| 函数名 `add` | `name` | 模型调用的工具名 |
| docstring | `description` | 告诉模型何时调用 |
| 类型标注 `a: int` | `inputSchema` | 校验参数并生成 JSON Schema |

`stdio` 传输不需要提前监听端口：Client 会把 `math_server.py` 启动为子进程，通过标准输入输出交换 MCP 消息。

> **stdio Server 不要用 `print()` 向标准输出写调试日志**，日志会混入协议数据。需要日志时写标准错误，或用 MCP 日志通知。

## 3. 不用模型，先验证链路

Agent 不调用工具，可能是模型选择、提示词、工具描述或协议连接中的任一环。最稳的排错顺序是先绕过模型，直接调用转换后的工具。

```python
import asyncio, sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER = Path(__file__).with_name("math_server.py").resolve()

def schema_as_dict(tool) -> dict:
    schema = tool.args_schema
    return schema if isinstance(schema, dict) else schema.model_json_schema()

def first_text(result):
    return next(block["text"] for block in result if block["type"] == "text")

async def main():
    client = MultiServerMCPClient(
        {"course_math": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)]}},
        tool_name_prefix=True,
    )
    tools = await client.get_tools()
    print("tools:", [t.name for t in tools])

    add_tool = next(t for t in tools if t.name == "course_math_add")
    print("description:", add_tool.description)
    print("required:", schema_as_dict(add_tool)["required"])
    print("37 + 58 =", first_text(await add_tool.ainvoke({"a": 37, "b": 58})))

asyncio.run(main())
```

关键输出：

```text
tools: ['course_math_add', 'course_math_multiply']
description: Add two integers exactly.
required: ['a', 'b']
37 + 58 = 95
```

这个检查点没用模型，却验证了六件事：依赖可导入、Client 能启动 stdio 子进程、MCP 初始化与工具发现成功、名称/描述/Schema 转换完成、LangChain Tool 能真实调用、Server 返回值能转成内容块。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_37-flowchart-mcp-validation-path.png)

<p>图 2：分层验证路径</p>

**结果为什么不是裸整数**：`add()` 返回 `95`，但 `ainvoke()` 返回标准内容块列表：

```python
[{"type": "text", "text": "95", "id": "..."}]
```

用 `first_text()` 提取文本，不依赖运行时生成的 `id`。

**为什么必须异步**：适配器把 MCP Tool 转成只有 coroutine、没有同步 `func` 的 `StructuredTool`。`add_tool.invoke(...)` 会报 `NotImplementedError: StructuredTool does not support sync invocation`。因此全程用异步：`await client.get_tools()`、`await tool.ainvoke(...)`、`await agent.ainvoke(...)`——`create_deep_agent()` 本身仍是同步的图构造函数。

## 4. 交给 Deep Agent

```python
import asyncio, os, sys
from pathlib import Path
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

SERVER = Path(__file__).with_name("math_server.py").resolve()

async def main():
    client = MultiServerMCPClient(
        {"course_math": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)]}},
        tool_name_prefix=True,
    )
    tools = await client.get_tools()

    model = ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
    )
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个严谨的计算助手。数学运算必须使用 MCP 工具完成，不要直接心算。",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "先计算 37 + 58，再把结果乘以 12。"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

合理工具序列：

```text
course_math_add(a=37, b=58) -> 95
course_math_multiply(a=95, b=12) -> 1140
```

最终措辞由模型决定，不应把某段回答当固定输出；真正要检查的是 Trace 或消息历史里是否出现两个 MCP 调用，且第二次用了第一次结果。

## 5. 多 Server 与传输

```python
client = MultiServerMCPClient(
    {
        "course_math": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)]},
        "langchain_docs": {"transport": "http", "url": "https://docs.langchain.com/mcp"},
    },
    tool_name_prefix=True,
)

all_tools = await client.get_tools()
math_tools = await client.get_tools(server_name="course_math")
```

| 传输 | 配置 | 生命周期 | 适用 |
|---|---|---|---|
| `stdio` | `command`、`args`、可选 `env` | Client 启动本地子进程 | 本地开发、可信命令、桌面应用 |
| `http` | `url`、可选 `headers`/`auth` | 连接已运行的 Streamable HTTP 服务 | 远程服务、团队共享、生产 |

新项目优先 Streamable HTTP；旧式 `sse` 已被规范弃用，WebSocket 也不是推荐主路径。

**Header 与认证**：静态 Bearer Token 放 `headers`，复杂 OAuth 用兼容的 `httpx.Auth`。**不要把 Token 设计成模型可填写的工具参数**，凭证应由应用配置或 Client 侧 Interceptor 注入。

**命名冲突**：适配器默认保留原始工具名，且不会自动去重。两个 Server 都有 `search`、或 MCP 工具与内置工具同名，会让模型与审批策略难以判断目标。启用 `tool_name_prefix=True` 后名字变成 `<server>_<tool>`，例如 `course_math_add`——从第一个示例就开启，新增 Server 时不必重命名已有审批规则。

## 6. 无状态调用与持久会话

`MultiServerMCPClient` 默认无状态：`get_tools()` 用临时 Session 发现，之后每次调用重新建 Session，执行完清理。对 stdio 意味着：

```text
启动子进程 -> 初始化 Session -> 调用工具 -> 关闭 Session -> 结束子进程
```

隔离清晰，适合查询类工具；但若 Server 把登录状态、游标或事务放进程内存，下一次调用不继承。

**显式保持 Session**：

```python
from langchain_mcp_adapters.tools import load_mcp_tools

async with client.session("course_math") as session:
    tools = await load_mcp_tools(
        session,
        callbacks=client.callbacks,
        tool_interceptors=client.tool_interceptors,
        server_name="course_math",
        tool_name_prefix=client.tool_name_prefix,
        handle_tool_errors=client.handle_tool_errors,
    )
    agent = create_deep_agent(model=model, tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "计算 20 + 22"}]})
```

Agent 的所有运行都必须在 `async with` 内完成；退出后 Session 关闭，基于它的工具不能继续执行。三个细节：`MultiServerMCPClient` 本身不是异步上下文管理器；直接 `load_mcp_tools(session)` 不继承回调、Interceptor、前缀与错误策略，需显式传；已打开的持久 Session 不会因 Interceptor 改 Header 而重新认证。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_38-comparison-mcp-session-lifecycle.png)

<p>图 3：Session 生命周期对比</p>

**Session 与 LangGraph Checkpoint 是两套生命周期**：Checkpoint 保存 Agent 状态，不会序列化正在运行的 MCP 子进程或网络连接。长时间暂停、进程重启、HITL 恢复不能依赖 Session 内存仍存在。重要状态应持久化在 Server 侧数据库，写操作尽量幂等；Session 只用于复用连接或短期上下文。

## 7. Resources、Prompts 与结构化结果

```python
blobs = await client.get_resources("knowledge", uris=["file:///handbook/returns.md"])
for blob in blobs:
    print(blob.metadata["uri"], blob.as_string())
```

Resources 转成 `Blob`，应用可写入虚拟文件系统、放提示词或交给检索——Client 不替应用选择。

```python
messages = await client.get_prompt("support", "triage", arguments={"severity": "high"})
```

Prompts 返回消息列表，不会自动安装成系统提示词。0.3.x 限制：Prompt 转换只接受文本，非文本会抛 `ValueError`；`get_resources(..., uris=None)` 只枚举静态 Resources，不展开需要参数的 Templates。

**结构化结果不会自动暴露给模型**：MCP Tool 可同时返回可读文本与 `structuredContent`；适配器把结构化部分放 `ToolMessage.artifact["structured_content"]`，默认不追加到模型文本。模型也需要看结构化字段时，在 Client 侧 Interceptor 有选择地转换——不要无条件把大型 JSON 复制进对话历史。

## 8. 错误语义与 Interceptor

适配器 0.3 起，Server 返回 `CallToolResult(isError=True)` 时默认转成 `status="error"` 的 `ToolMessage`，Agent 可读错误后修参数或换工具，不必终止整次运行。

| 失败类型 | 默认行为 |
|---|---|
| `CallToolResult(isError=True)` | 返回失败 ToolMessage |
| 传输失败 / Session 初始化失败 / 内容转换失败 | 抛异常 |

希望工具执行错误也走异常路径时，`MultiServerMCPClient(..., handle_tool_errors=False)`，`isError=True` 会以 `ToolException` 抛出。

> **这套错误语义不适用于搜索完整性**：v0.7 的 `grep`/`glob` 可以成功返回部分结果并标 `truncated=True`；`ToolMessage` 没报错只证明调用成功，不证明结果完整。

**Interceptor ≠ Agent Middleware**：Interceptor 包装「调用 MCP Tool」这一步，适合注入用户/租户信息、加 Header、记审计、调用前拒绝、转换结果；Deep Agents Middleware 包装更广的模型与工具循环。

```python
async def require_authentication(request: MCPToolCallRequest, handler):
    if (request.server_name == "orders" and request.name == "cancel"
            and not request.runtime.state.get("authenticated", False)):
        return ToolMessage(content="Authentication required.",
                           tool_call_id=request.runtime.tool_call_id, status="error")
    return await handler(request)
```

`request.name` 是 Server 原始工具名，`server_name` 判断来源；这与 `interrupt_on` 使用最终带前缀的 LangChain Tool 名不同。Server 在独立进程，读不到 LangGraph State/Store/Context；Interceptor 是 Client 侧桥梁，但真正的授权与参数校验仍应在 Server 侧执行。

## 9. 与安全机制组合

**文件权限不覆盖 MCP**：第 12 篇的 `FilesystemPermission` 只控制内置文件工具；即使禁止内置 `read_file` 访问敏感路径，一个 MCP 文件 Server 仍可能用自己的进程权限读取同一路径。

| 入口 | 控制方式 |
|---|---|
| 内置文件工具 | `FilesystemPermission` |
| MCP Tool | Server 端 ACL、参数校验、进程隔离、Interceptor、HITL |
| 沙箱 `execute` | 沙箱文件、命令与网络策略 |

**副作用工具配 HITL**：MCP 工具转成标准 Tool 后可用 `interrupt_on`，但配置键必须是模型最终看到的工具名——启用前缀后写 `billing_charge_card`，不是原始 `charge_card`。审批发生在适配器真正调用 Server 之前；MCP annotation 的 `destructiveHint` 会保留在 metadata，但不会自动生成审批策略。

**收缩子 Agent 工具**：默认 general-purpose 继承父 Agent 的额外工具，声明式子 Agent 省略 `tools` 时也继承；显式 `tools` 会整体替换该组额外工具（不是追加），但 Harness 文件工具与可能的 `execute` 不会自动移除。主 Agent 的 `TodoListMiddleware` 也不会传给子 Agent。

```python
agent = create_deep_agent(
    model=model,
    tools=all_mcp_tools,
    subagents=[{
        "name": "catalog-reader",
        "description": "只使用只读 MCP 工具查询商品目录",
        "system_prompt": "只使用目录查询 MCP 工具回答商品问题。",
        "tools": read_only_mcp_tools,
        "permissions": [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
    }],
)
```

声明式 `SubAgent` 省略 `interrupt_on` 会继承父映射；一旦提供就整体替换，必须重复仍需保留的规则。`CompiledSubAgent` 与远程 `AsyncSubAgent` 都不继承父审批，应分别配置。

**stdio 是本地代码执行配置**：`command`、`args`、`env` 会用当前应用用户权限执行，SDK 不会弹「是否信任」确认框。上线前：只运行审核过的包与命令、固定依赖版本与来源、用绝对路径、最小化环境变量、Server 内再次校验路径/租户/资源、Web 场景优先隔离的远程 MCP 服务。远程 HTTP 同样不是天然安全：用 TLS、限制可连主机、防止访问 localhost 或云元数据地址、审计 Header 中的秘密。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_39-framework-mcp-security-boundaries.png)

<p>图 4：MCP 工具的安全控制边界</p>

## 10. 排错清单

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `StructuredTool does not support sync invocation` | 用了 `invoke()` | 改 `ainvoke()`，Agent 也走异步 |
| 找不到 `math_server.py` | stdio 子进程工作目录不同 | 用 `Path(__file__).with_name(...).resolve()` |
| Server 一启动就协议解析失败 | stdout 打印了调试日志 | 日志写 stderr 或用 MCP 日志通知 |
| 工具名不是 `course_math_add` | 没开前缀 / 持久 Session 漏传选项 | `tool_name_prefix=True`，`load_mcp_tools` 显式重复 |
| 每次调用后 Server 状态丢失 | 默认每次新建 Session | 用 `client.session()` 或 Server 侧持久化 |
| 模型不调用 MCP Tool | 模型工具调用不稳或描述不清 | 先直接 `ainvoke()` 验证工具，再查模型与提示词 |
| 工具报错但没抛异常 | 0.3 默认把 `isError` 返给模型 | 看 `ToolMessage.status`，或 `handle_tool_errors=False` |
| HTTP 401/403 | Header/Token/OAuth/ACL 错 | 先用独立 Client 验证认证，不让模型填凭证 |
| HITL 没拦截 | `interrupt_on` 用了原始名 | 改为带前缀的最终工具名 |
| 文件权限没拦 MCP | 权限不覆盖 MCP Tool | 在 Server、进程边界与 HITL 分别控制 |

排错按层次缩小范围：**Server 函数 → 工具发现 → LangChain Tool 直接调用 → Deep Agent 调用 → 多 Server/认证/HITL/子 Agent**。不要一开始就把模型、认证、持久会话与多 Agent 同时接入；无密钥冒烟测试越早通过，后续越容易定位。

## 11. 本篇验收

- [ ] 能说清 MCP 与 Deep Agents 的分层，知道 `create_deep_agent` 不管 Session；
- [ ] 会写一个 FastMCP stdio Server，并避免 stdout 日志污染协议；
- [ ] 能先不用模型验证工具发现、Schema 与真实调用；
- [ ] 知道 MCP 工具必须异步调用；
- [ ] 会用 `tool_name_prefix` 避免命名冲突；
- [ ] 理解默认无状态 Session 与 `client.session()` 的差异；
- [ ] 知道 Resources/Prompts 不会自动进入上下文；
- [ ] 能配 Interceptor 与 HITL（带前缀的工具名）；
- [ ] 记住文件权限不覆盖 MCP，stdio 是本地代码执行入口。

> **补记**：MCP 的价值是「协议统一」，不是「自动安全」。最实用的接入顺序是**先无密钥验证工具链，再接模型**；最容易忽略的边界是命名与生命周期——工具名加 Server 前缀既防冲突又决定 `interrupt_on` 的写法，而 Session 不是持久化，重要状态必须落到 Server 数据库。

## 写在最后

下一篇[评分量规](14-评分量规)把「完成」写成可取证的标准：工作模型生成、评分模型取证、失败回灌修订，只有 `satisfied` 才放行。
