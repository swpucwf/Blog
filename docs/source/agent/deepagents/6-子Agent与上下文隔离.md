# 子 Agent 与上下文隔离

## 1. 上下文膨胀问题

主 Agent 要写研究报告，其中一个子任务是「搜索 LangGraph 技术文档」。这个子任务可能包含：5 次网络搜索、每次返回 3000+ tokens、多次写文件保存中间结果、多次读文件回顾整理。

这些中间过程全堆在主 Agent 上下文里。自动卸载能缓解一部分，但主 Agent **根本不需要知道这些细节**——它只需要最终摘要。

## 2. Context Quarantine：上下文隔离

子 Agent 的核心设计动机就是 **Context Quarantine**：

1. 主 Agent 通过 `task` 工具**创建**子 Agent；
2. 子 Agent 在**独立上下文**中执行（中间调用记录不进入主对话；共用 Backend 时文件仍可共享）；
3. 完成后只把**最终结果**返回主 Agent；
4. 主 Agent 的上下文保持干净。

打个比方：主 Agent 是项目经理，子 Agent 是专项负责人。项目经理不需要参加每一场技术讨论，只看负责人提交的总结报告。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_13-comparison-context-quarantine.png

<p>图 1：有无子 Agent 的上下文对比</p>

**什么时候用**：

| 场景 | 用子 Agent？ | 理由 |
|---|---|---|
| 多次搜索与整理的研究任务 | 是 | 大量中间结果会膨胀主上下文 |
| 需要特殊工具或指令的专业任务 | 是 | 子 Agent 可有自己的工具集与提示词 |
| 需要不同模型能力 | 是 | 子 Agent 可用不同模型 |
| 需要高层协调的复杂任务 | 是 | 主 Agent 协调，子 Agent 执行 |
| 单步简单查询 | 否 | 委派开销大于收益 |
| 需要主 Agent 看到完整调用过程 | 否 | 主 Agent 只收到最终结果 |

## 3. 用字典定义子 Agent

最常用的方式是字典：

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "researcher",                # 必填：唯一标识
    "description": "深入研究特定主题，搜索多个信息源并整理成摘要",  # 必填：主 Agent 据此路由
    "system_prompt": """你是一位专业的研究员。你的任务是：
1. 把研究问题拆解为多个搜索查询
2. 用 internet_search 搜索相关信息
3. 整理发现，写成简洁摘要
4. 列出关键发现和信息来源

注意：返回结果控制在 500 字以内，只返回核心发现。""",
    "tools": [internet_search],          # 可选，默认继承；显式指定后完全替换
    "skills": ["/skills/research/"],     # 可选，不继承主 Agent
}

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    skills=["/skills/main/"],            # 主 Agent 与 general-purpose 继承
    subagents=[research_subagent],       # researcher 只获得 /skills/research/
)
```

**字段与继承规则**：

| 字段 | 必填 | 继承主 Agent？ | 说明 |
|---|---|---|---|
| `name` | 是 | — | 唯一标识；同时作为流式输出与追踪的 `lc_agent_name` |
| `description` | 是 | — | 能力描述，主 Agent 据此决策路由 |
| `system_prompt` | 是 | **否** | 子 Agent 自己的指令，必须独立定义 |
| `tools` | 否 | 默认继承，指定后**完全替换** | 工具集，不合并 |
| `model` | 否 | 默认继承 | 支持 LangChain 对象或 `"provider:model"` 字符串 |
| `middleware` | 否 | 否 | 子 Agent 自己的中间件 |
| `interrupt_on` | 否 | 默认继承 | 可覆盖主 Agent 的审批配置 |
| `skills` | 否 | 否 | 指定后运行独立 `SkillsMiddleware`，状态与主 Agent 隔离 |
| `response_format` | 否 | 否 | 结构化输出 schema（需 `deepagents>=0.5.3`） |
| `permissions` | 否 | 默认继承，指定后**完全替换** | 文件权限规则，整体替换而非叠加 |

> **v0.7**：默认 `general-purpose` 与声明式子 Agent **都不继承** 主 Agent 的 `TodoListMiddleware`；需要规划就在子 Agent 自己的 `middleware` 里加。默认 `general-purpose` 要用同名 spec 覆盖。

## 4. 默认的 general-purpose

即使不定义任何子 Agent，也自带一个 **general-purpose**：继承主 Agent 显式传入的 `tools`、`model` 与 `skills`，使用自己的默认系统提示词（`DEFAULT_SUBAGENT_PROMPT`）并应用模型的 Harness Profile 配置，**不继承主 Agent 的 `system_prompt`**（这一行为始于 0.0.6，不是 v0.7 新增）。

```python
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt="你是一位研究助手。",   # 只配置主 Agent
)
# 主 Agent 可以: task(subagent_type="general-purpose", description="搜索量子计算最新进展")
```

它复用主 Agent 的模型与显式工具，用独立上下文处理委派，只回一份精炼摘要。

**禁用**：不想让 Agent 拥有 `task` 工具时，两步缺一不可：

```python
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

register_harness_profile(
    key="openai:zai-org/GLM-5.2",
    profile=HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    ),
)

agent = create_deep_agent(model=model, subagents=[])
```

> 不要用 `excluded_middleware` 排除 `SubAgentMiddleware`——会直接抛 `ValueError`。正确做法是 `enabled=False`。

**覆盖**：用 `name="general-purpose"` 同名 spec 自定义提示词、模型或工具：

```python
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    subagents=[
        {
            "name": "general-purpose",
            "description": "通用助手，处理各种委派任务",
            "system_prompt": "你是一个通用助手。",
            "tools": [internet_search],
            "model": ChatOpenAI(model="zai-org/GLM-5.2", api_key=..., base_url=...),
        },
    ],
)
```

**为它启用 Todo**：用同名 spec 显式配中间件；主、子两处配置相互独立，子 Agent 不会读取或写回主 Agent 的清单：

```python
agent = create_deep_agent(
    model=model,
    middleware=[TodoListMiddleware()],
    subagents=[
        {
            "name": "general-purpose",
            "description": "处理需要规划的委派任务",
            "system_prompt": "为委派任务制定计划，执行后返回结果。",
            "middleware": [TodoListMiddleware()],
        },
    ],
)
```

## 5. CompiledSubAgent：复用 LangGraph 工作流

子任务需要多步骤、有分支逻辑时，可以把预构建的 LangGraph 图作为子 Agent：

```python
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent

custom_graph = create_agent(
    model=model,
    tools=[statistical_analysis, generate_chart],
    system_prompt="你是数据分析专家，擅长统计分析和可视化。",
)

data_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="执行复杂的数据分析任务，包括统计分析和图表生成",
    runnable=custom_graph,
)

agent = create_deep_agent(model=model, subagents=[data_subagent])
```

要求：传入图的 State 必须包含 `"messages"` 键。

| 场景 | 方式 | 理由 |
|---|---|---|
| 大多数情况 | 字典 | 简单直观、配置灵活 |
| 需要复杂多步骤工作流 | CompiledSubAgent | 可用图 API 定义分支、循环 |
| 已有现成 LangGraph 图 | CompiledSubAgent | 直接复用，无需重写 |

## 6. 多子 Agent 协作

常见模式：多个专业子 Agent 分工，主 Agent 作为协调者。

```python
subagents = [
    {
        "name": "data-collector",
        "description": "从多个来源收集原始数据，包括网络搜索和 API 调用",
        "system_prompt": "你是数据收集专家。搜索并整理相关数据，返回结构化摘要。",
        "tools": [internet_search, api_call],
    },
    {
        "name": "data-analyzer",
        "description": "对收集到的数据统计分析，提取关键洞察",
        "system_prompt": "你是数据分析专家。分析数据并提取 3-5 个关键发现，控制在 300 字以内。",
        "tools": [statistical_analysis],
    },
    {
        "name": "report-writer",
        "description": "根据分析结果撰写专业报告",
        "system_prompt": "你是技术写作专家。根据分析结果撰写清晰、专业的报告。",
        "tools": [format_document],
    },
]

agent = create_deep_agent(
    model=model,
    middleware=[TodoListMiddleware()],
    system_prompt="""你是一位项目协调者。面对复杂任务时：
1. 先用 write_todos 制定计划
2. 将数据收集委派给 data-collector
3. 将分析工作委派给 data-analyzer
4. 将报告撰写委派给 report-writer
5. 整合各子 Agent 的输出，形成最终结果""",
    subagents=subagents,
)
```

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_14-framework-multi-subagent.png

<p>图 2：多子 Agent 协作模式</p>

执行流程：制定计划 → `task(data-collector)` → `task(data-analyzer)` → `task(report-writer)` → 主 Agent 整合。每一步子 Agent 都在独立上下文工作，主 Agent 只见精炼结果。

## 7. 结构化输出

默认主 Agent 收到子 Agent 最后一条消息的自由文本。用 `response_format` 可以让子 Agent 返回符合 Pydantic schema 的 JSON：

```python
from pydantic import BaseModel, Field

class ResearchFindings(BaseModel):
    summary: str = Field(description="研究摘要")
    confidence: float = Field(description="置信度 0-1")
    sources: list[str] = Field(description="信息来源 URL 列表")

research_subagent = {
    "name": "researcher",
    "description": "研究特定主题并返回结构化发现",
    "system_prompt": "深入研究给定主题，返回你的发现。",
    "tools": [internet_search],
    "response_format": ResearchFindings,   # 需 deepagents>=0.5.3
}
```

设置后主 Agent 始终收到符合 schema 的有效 JSON，便于程序化处理。

## 8. 最佳实践

1. **描述要具体、行为导向**：主 Agent 靠 `description` 决定何时委派。「执行深度网络研究，需要多次搜索、交叉验证和综合分析时使用」好于「做研究」。
2. **系统提示词要详细**：包含工具使用指导、输出格式、字数限制——直接影响回给主 Agent 的质量。
3. **工具集要精简**：最小权限，只给需要的工具（研究 Agent 就别给发邮件、删文件、执行代码）。
4. **不同子 Agent 用不同模型**：快速查询用轻量模型，深度分析用强推理模型。
5. **返回结果要精简**：明确「只回核心摘要，不含原始数据、中间计算、工具输出」，否则上下文隔离就失去意义。

## 9. 排查

| 现象 | 原因 | 解法 |
|---|---|---|
| 子 Agent 没被调用，主 Agent 全做了 | description 无法让主 Agent 判断何时委派 | 描述更具体；在主 Agent 提示词中明确要求用 `task()` |
| 上下文依然膨胀 | 子 Agent 回了大量原始数据 | 强制简洁返回；让子 Agent 把原始数据写文件，只回分析摘要 |
| 调用了错误的子 Agent | 多个 description 太相似 | 明确区分使用场景（快速查询 vs 深度研究） |

## 10. 本篇验收

- [ ] 能解释 Context Quarantine 与「主 Agent 只收摘要」；
- [ ] 能判断什么任务该委派、什么任务不值得；
- [ ] 记住必填三字段与 `tools/skills/permissions` 的继承/替换规则；
- [ ] 知道 general-purpose 不继承主 Agent 的 `system_prompt` 与 Todo；
- [ ] 能用同名 spec 覆盖或禁用 general-purpose；
- [ ] 会用 `CompiledSubAgent` 复用 LangGraph 图；
- [ ] 能用 `response_format` 让子 Agent 返回 JSON；
- [ ] 掌握五条最佳实践与三类常见故障的排查。

> **补记**：子 Agent 的价值不是「多一个 Agent」，而是**把中间过程挡在主上下文之外**。因此设计时最该盯住两件事：`description` 决定路由是否准确，`system_prompt` 里的「返回什么、返回多少、不要返回什么」决定隔离是否真的成立。工具与权限默认继承、显式指定即整体替换，这一点在给子 Agent 收权时尤其容易踩坑。

## 写在最后

下一篇[异步子 Agent](7-异步子Agent)解决同步委派的两个瓶颈：主 Agent 被阻塞、任务无法中途追加或取消。
