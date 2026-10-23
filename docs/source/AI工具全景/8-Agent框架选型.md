# Agent 框架选型

## 1. 先看五种设计模式

选框架之前要先确定模式，否则容易「用重框架做轻任务」：

| 模式 | 核心思想 | 适用场景 | 代表实现 |
|---|---|---|---|
| ReAct | 推理与行动交替，观察结果再决定下一步 | 需要工具调用的单 Agent | LangChain ReAct |
| Plan-Execute | 先规划完整步骤，再逐步执行并可调整 | 多步骤复杂任务 | LangGraph PlanExecute |
| Multi-Agent 编排 | 多个 Agent 分工，通过对话或消息协调 | 多角色协作 | CrewAI、AutoGen、LangGraph |
| 反射（Reflection） | 执行后自评，发现问题就修正重做 | 质量要求高的输出 | LangGraph Self-Reflective |
| 路由（Router） | 按意图分发给不同专业 Agent | 多领域问答、客服 | LangChain Router |

对应建议：单 Agent 加工具用 ReAct；步骤多用 Plan-Execute；多角色用 Multi-Agent；质量优先用 ReAct + Reflection。设计模式本身的取舍，见[Agent 深入理解](/source/agent/深入理解/README)。

## 2. 主流框架

| 框架 | 特点 |
|---|---|
| LangChain / LangGraph | 生态最成熟：LangChain 与 LangGraph 均已在 2025-10 GA，核心抽象为 `create_agent`；LangGraph 专注状态图多 Agent 编排，生产验证最充分 |
| Microsoft Agent Framework | 统一 AutoGen 与 Semantic Kernel 的多 Agent 模式，支持 Python 与 .NET，2026-04 发布 1.0 |
| AutoGen | 微软早期多 Agent 对话框架，**已进入维护模式**，新项目建议改用 MAF；社区分支 AG2 仍在演进 |
| CrewAI | 角色化多 Agent 协作，上手最快 |
| Claude Code | Anthropic 的 Agent 能力封装，编码场景强 |
| OpenClaw | 开源本地智能体，强调动手执行与 7×24 自动化 |
| Hermes Agent | 开源自托管框架，主打持久记忆、跨会话沉淀技能、多平台接入 |

## 3. 横向对比

| 框架 | 成熟度 | 学习曲线 | 多 Agent | 记忆 | 适用规模 |
|---|---|---|---|---|---|
| LangChain / LangGraph | 极高 | 中 | 状态图编排 | 需自行集成 | 生产级 |
| Microsoft Agent Framework | 中 | 中 | 融合 AutoGen + SK | — | 企业级 |
| AutoGen | 高 | 中 | 对话式 | 有限 | 中型以上（维护模式） |
| CrewAI | 高 | 低 | 角色化分工 | 有限 | 中小型 |
| Hermes Agent | 中 | 低 | 支持 | 持久记忆 + 自进化 | 中型 |
| OpenClaw | 中 | 低 | Skills 驱动 | 会话级 | 中小型 |
| Claude Code | 极高 | 低 | 内建 Agent 模式 | 会话级 | 生产级编码 |

## 4. 决策树

```text
需要编码 Agent？
 ├── 是 → Claude Code（终端）/ OpenClaw（本地自动化）
 └── 否 → 需要多 Agent 协作？
           ├── 是 → 生产级？
           │        ├── 是 → LangGraph / Microsoft Agent Framework
           │        └── 否 → CrewAI（快速上手）/ MAF（微软生态）
           └── 否 → 需要持久记忆？
                    ├── 是 → Hermes Agent
                    └── 否 → 低代码平台（Coze / Dify）
```

## 5. CrewAI 五分钟示例

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="研究员", goal="研究指定主题的最新进展",
                   backstory="你是一位资深 AI 研究员")
writer = Agent(role="撰稿人", goal="将研究结果写成简洁的文章",
               backstory="你是一位擅长科技写作的撰稿人")

research = Task(description="研究 2026 年 RAG 技术的最新进展", agent=researcher)
write = Task(description="基于研究结果，写一篇 500 字的技术摘要", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research, write])
print(crew.kickoff())
```

国内使用可以把 `openai_api_base` 指向 DeepSeek 等国内端点，降低成本。

## 6. 低代码平台

- **Coze**：零代码工作流，适合 AI 密集型任务与快速验证；
- **Dify**：低代码 AI 应用工厂，支持私有化部署，适合企业级复杂流程；
- **n8n**：开源流程自动化中枢，强在系统集成与高度定制。

三者的分界线是「流程确定性」：流程固定、参与者多、需要审计，用低代码更快；流程需要动态决策，回到代码框架。

## 7. 选型时最该问的三个问题

- **状态复杂度**：单轮还是多轮、是否需要跨会话记忆？要持久记忆就排除会话级框架。
- **验收方式**：靠测试、规则，还是人工评审？没有机器可验的标准，多 Agent 并行的收益会被评审成本吃掉。
- **团队栈**：Python 还是 .NET？生态匹配比框架排名更重要。

## 写在最后

框架解决「agent 怎么搭」，下一个问题是「知识从哪来」。下一篇[RAG 框架选型](9-RAG框架选型)对比 LlamaIndex、RAGFlow 等实现路径与演进方向。
